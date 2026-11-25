# FlashInfer's min-p sampling, top-p / top-k renorm, top-k mask-logits and softmax(temperature)
# on the shapes and data of cute-rs's `sampling` example (vocab 128512, batches 1, 16, 128, 512;
# softmax of normal(1) logits, built from the same SplitMix64 hash). CUDA-graph replays: 100
# launches per graph, one replay between two CUDA events, best of 5.
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


def logits(kind, param, n):
    i = np.arange(n, dtype=np.uint64)
    u1 = hash_uniform(1, i)
    u2 = hash_uniform(2, i)
    return param * np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)


def probs_of(kind, param, batch, d):
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
d = 128512
batches = (1, 16, 128, 512)
SEED, OFFSET = 42, 0
allp = probs_of("normal", 1.0, batches[-1], d)
alll = torch.from_numpy(logits("normal", 1.0, batches[-1] * d).reshape(batches[-1], d)).float().cuda()
BENCH = [
    ("min_p", 0.1, "p", lambda p, l: S.min_p_sampling_from_probs(p, 0.1, seed=SEED, offset=OFFSET)),
    ("min_p", 0.5, "p", lambda p, l: S.min_p_sampling_from_probs(p, 0.5, seed=SEED, offset=OFFSET)),
    ("top_p_renorm", 0.5, "p", lambda p, l: S.top_p_renorm_probs(p, 0.5)),
    ("top_p_renorm", 0.9, "p", lambda p, l: S.top_p_renorm_probs(p, 0.9)),
    ("top_k_renorm", 10, "p", lambda p, l: S.top_k_renorm_probs(p, 10)),
    ("top_k_renorm", 1000, "p", lambda p, l: S.top_k_renorm_probs(p, 1000)),
    ("top_k_mask", 10, "l", lambda p, l: S.top_k_mask_logits(l, 10)),
    ("top_k_mask", 1000, "l", lambda p, l: S.top_k_mask_logits(l, 1000)),
    ("softmax", 0.7, "l", lambda p, l: S.softmax(l, 0.7)),
    ("softmax", 1.0, "l", lambda p, l: S.softmax(l, 1.0)),
]
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
print("kernel        param  batch  time(us)")
for name, param, src, fn in BENCH:
    for b in batches:
        p, l = allp[:b].contiguous(), alll[:b].contiguous()
        t = graph_bench(lambda: fn(p, l))
        print(f"{name:13} {param:5}  {b:5}  {t:8.2f}")
