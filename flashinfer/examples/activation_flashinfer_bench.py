# FlashInfer silu_and_mul / gelu_and_mul / gelu_tanh_and_mul on the shapes of cute-rs's
# `activation` example (tokens {1, 64, 4096} x d {4096, 14336}), timed from CUDA-graph replays
# of one L2-resident buffer pair: 100 launches per graph, one replay between two CUDA events.
import torch
import flashinfer

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
print("act        dtype  tokens  d      time(us)    GB/s")
for act in ("silu", "gelu", "gelu_tanh"):
    fn = getattr(flashinfer, f"{act}_and_mul")
    for dt, name in ((torch.float16, "f16"), (torch.bfloat16, "bf16")):
        for d in (4096, 14336):
            for tokens in (1, 64, 4096):
                x = torch.randn(tokens, 2 * d, dtype=dt, device=dev) * 2
                out = torch.empty(tokens, d, dtype=dt, device=dev)
                t = graph_bench(lambda: fn(x, out))
                nbytes = tokens * 3 * d * 2
                print(f"{act:9}  {name:5}  {tokens:6}  {d:5}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")
