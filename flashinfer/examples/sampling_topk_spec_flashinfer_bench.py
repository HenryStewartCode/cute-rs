# FlashInfer's `top_k` (values and indices) and `chain_speculative_sampling` on the shapes of
# cute-rs's `sampling` example: vocab 128512, batches 1/16/128/512, randn(1) scores as in
# benchmarks/bench_topk.py, k in {50, 256, 1000, 2048}; speculative: 1 or 4 draft tokens with
# normal(1) draft and target rows (draft ids drawn with torch.multinomial, so the accepted
# prefixes differ from cute-rs's but the memory traffic is the same). Timed from CUDA-graph
# replays: 100 launches per graph, one replay between two CUDA events, best of 5. Explicit
# seed/offset and preallocated output counters keep host-side work out of the graph.
import numpy as np
import torch
import flashinfer

dev = "cuda"
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix(x):
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_uniform(stream, i):
    x = np.uint64(stream) * np.uint64(0x100000001) + i
    return ((splitmix(x) >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def logits(kind, param, n):
    i = np.arange(n, dtype=np.uint64)
    u1 = hash_uniform(1, i)
    if kind == "normal":
        u2 = hash_uniform(2, i)
        return param * np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return np.log(-np.log(u1 + 1e-20) + 1e-20) / param


def probs_of(kind, param, batch, d):
    with np.errstate(over="ignore"):
        x = torch.from_numpy(logits(kind, param, batch * d).reshape(batch, d))
    return torch.softmax(x.to(dev), dim=-1).float().contiguous()


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



S = flashinfer.sampling
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
d = 128512
batches = (1, 16, 128, 512)
allp = probs_of("normal", 1.0, batches[-1], d)
scores = torch.from_numpy(logits("normal", 1.0, batches[-1] * d).reshape(batches[-1], d)).to(dev).float().contiguous()
print("kernel        k      batch  time(us)")
for k in (50, 256, 1000, 2048):
    for b in batches:
        s = scores[:b].contiguous()
        t = graph_bench(lambda: flashinfer.top_k(s, k))
        print(f"{'top_k':13} {k:5}  {b:5}  {t:8.2f}")
print("kernel                  n  batch  time(us)")
for n in (1, 4):
    for b in batches:
        draft = probs_of("normal", 1.0, b * n, d).reshape(b, n, d).contiguous()
        target = probs_of("normal", 1.0, b * (n + 1), d).reshape(b, n + 1, d).contiguous()
        ids = torch.multinomial(draft.reshape(b * n, d), 1).reshape(b, n).int().contiguous()
        acc = torch.zeros(b, dtype=torch.int32, device=dev)
        emi = torch.zeros(b, dtype=torch.int32, device=dev)
        t = graph_bench(
            lambda: S.chain_speculative_sampling(draft, ids, target, acc, emi, True, None, seed=42, offset=0)
        )
        print(f"{'chain_speculative':20} {n:3}  {b:5}  {t:8.2f}")
