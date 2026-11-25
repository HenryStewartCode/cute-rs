# FlashInfer rmsnorm / fused_add_rmsnorm on the shapes of cute-rs's `norm` example
# (benchmarks/bench_fused_add_rmsnorm.py's batches and hidden sizes), timed from CUDA-graph
# replays: 100 launches per graph, one replay between two CUDA events.
import torch
import flashinfer

dev = "cuda"
EPS = 1e-6


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
# Correctness against PyTorch.
for dt in (torch.float16, torch.bfloat16):
    x = torch.randn(19, 3072, dtype=dt, device=dev)
    r = torch.randn_like(x)
    w = torch.randn(3072, dtype=dt, device=dev)
    ref = lambda v: (v.float() * torch.rsqrt(v.float().pow(2).mean(-1, keepdim=True) + EPS) * w.float())
    e1 = (flashinfer.rmsnorm(x, w, EPS).float() - ref(x)).abs().max().item()
    xs = x.clone(); rs = r.clone()
    flashinfer.fused_add_rmsnorm(xs, rs, w, EPS)
    e2 = (xs.float() - ref(x.float() + r.float())).abs().max().item()
    print(f"{dt}: rmsnorm err {e1:.2e}, fused err {e2:.2e}")

print("variant    dtype  batch  hidden  time(us)    GB/s")
for variant in ("Rms", "FusedAdd", "GemmaRms", "GemmaFusedAdd", "LayerNorm"):
    for dt, name in ((torch.float16, "f16"), (torch.bfloat16, "bf16")):
        if variant == "LayerNorm" and dt != torch.bfloat16:
            continue  # FlashInfer's layernorm takes bf16 input only
        for batch in (1, 19, 99, 989):
            for d in (3072, 4096, 8192):
                x = torch.randn(batch, d, dtype=dt, device=dev)
                r = torch.randn_like(x)
                w = torch.randn(d, dtype=dt, device=dev)
                out = torch.empty_like(x)
                if variant == "Rms":
                    fn = lambda: flashinfer.rmsnorm(x, w, EPS, out=out)
                    nbytes = batch * d * 2 * 2 + d * 2
                elif variant == "GemmaRms":
                    fn = lambda: flashinfer.gemma_rmsnorm(x, w, EPS, out=out)
                    nbytes = batch * d * 2 * 2 + d * 2
                elif variant == "GemmaFusedAdd":
                    fn = lambda: flashinfer.gemma_fused_add_rmsnorm(x, r, w, EPS)
                    nbytes = batch * d * 2 * 4 + d * 2
                elif variant == "LayerNorm":
                    g = torch.randn(d, dtype=torch.float32, device=dev)
                    b = torch.randn(d, dtype=torch.float32, device=dev)
                    fn = lambda: flashinfer.layernorm(x, g, b, EPS)
                    nbytes = batch * d * 2 * 2 + d * 8
                else:
                    fn = lambda: flashinfer.fused_add_rmsnorm(x, r, w, EPS)
                    nbytes = batch * d * 2 * 4 + d * 2
                t = graph_bench(fn)
                print(f"{variant:9}  {name:5}  {batch:5}  {d:6}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")

# rmsnorm_quant / fused_add_rmsnorm_quant (E4M3 out, scale 0.02) on the same shapes.
scale = torch.tensor([0.02], dtype=torch.float32, device=dev)
for variant in ("RmsQuant", "FusedAddQuant"):
    for dt, name in ((torch.float16, "f16"), (torch.bfloat16, "bf16")):
        for batch in (1, 19, 99, 989):
            for d in (3072, 4096, 8192):
                x = torch.randn(batch, d, dtype=dt, device=dev)
                r = torch.randn_like(x)
                w = torch.randn(d, dtype=dt, device=dev)
                out = torch.empty(batch, d, dtype=torch.float8_e4m3fn, device=dev)
                if variant == "RmsQuant":
                    fn = lambda: flashinfer.norm.rmsnorm_quant(out, x, w, scale, EPS)
                    nbytes = batch * d * 3 + d * 2
                else:
                    fn = lambda: flashinfer.norm.fused_add_rmsnorm_quant(out, x, r, w, scale, EPS)
                    nbytes = batch * d * 5 + d * 2
                t = graph_bench(fn)
                print(f"{variant:9}  {name:5}  {batch:5}  {d:6}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")

# rmsnorm of a 3-D input (QKRMSNorm): a [batch, heads, 128] slice of packed QKV.
for dt, name in ((torch.float16, "f16"), (torch.bfloat16, "bf16")):
    for heads, d in ((32, 128), (8, 128), (64, 128)):
        for batch in (1, 19, 99, 989):
            qkv = torch.randn(batch, heads + 2, d, dtype=dt, device=dev)
            x = qkv[:, :heads, :]
            w = torch.randn(d, dtype=dt, device=dev)
            out = torch.empty(batch, heads, d, dtype=dt, device=dev)
            fn = lambda: flashinfer.rmsnorm(x, w, EPS, out=out)
            t = graph_bench(fn)
            nbytes = batch * heads * d * 4 + d * 2
            print(f"QkRms      {name:5}  {batch:5}  {heads:2}x{d}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")
