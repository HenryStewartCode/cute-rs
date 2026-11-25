# FlashInfer's sampling_from_logits and top_k_top_p_sampling_from_{probs,logits} (both
# filter_apply_orders) on the shapes and data of cute-rs's `sampling` example with SAMPLING_NEW=1
# (vocab 128256, batches 1, 16, 128; normal(1) logits from the SplitMix64 hash, k = 50, p = 0.9).
# CUDA-graph replays: 100 launches per graph, one replay between two CUDA events, best of 5.
import numpy as np
import torch
import flashinfer

dev = "cuda"


def splitmix(x):
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_uniform(stream, i):
    x = np.uint64(stream) * np.uint64(0x100000001) + i
    return ((splitmix(x) >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def logits(n):
    i = np.arange(n, dtype=np.uint64)
    u1, u2 = hash_uniform(1, i), hash_uniform(2, i)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)


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
d = 128256
batches = (1, 16, 128)
SEED, OFFSET = 42, 0
alll = torch.from_numpy(logits(batches[-1] * d).reshape(batches[-1], d)).float().to(dev)
allp = torch.softmax(alll, dim=-1).contiguous()
BENCH = [
    ("sampling_from_logits", lambda p, l, b: S.sampling_from_logits(l, seed=SEED, offset=OFFSET)),
    ("top_k_top_p_from_probs_Joint", lambda p, l, b: S.top_k_top_p_sampling_from_probs(p, 50, 0.9, filter_apply_order="joint", seed=SEED, offset=OFFSET)),
    ("top_k_top_p_from_probs_TopKFirst", lambda p, l, b: S.top_k_top_p_sampling_from_probs(p, 50, 0.9, filter_apply_order="top_k_first", seed=SEED, offset=OFFSET)),
    ("top_k_top_p_from_logits_Joint", lambda p, l, b: S.top_k_top_p_sampling_from_logits(l, 50, 0.9, filter_apply_order="joint", seed=SEED, offset=OFFSET)),
    ("top_k_top_p_from_logits_TopKFirst", lambda p, l, b: S.top_k_top_p_sampling_from_logits(l, 50, 0.9, filter_apply_order="top_k_first", seed=SEED, offset=OFFSET)),
]
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
print("kernel                           batch  time(us)")
for name, fn in BENCH:
    for b in batches:
        p, l = allp[:b].contiguous(), alll[:b].contiguous()
        t = graph_bench(lambda: fn(p, l, b))
        print(f"{name:32} {b:5}  {t:8.2f}")
