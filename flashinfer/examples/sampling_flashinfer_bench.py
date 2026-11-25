# FlashInfer's samplers on the shapes and data of cute-rs's `sampling` example: vocab
# 128512 and batches from benchmarks/bench_sampling.py, softmax of normal(1) and gumbel(0.1)
# logits built from the same SplitMix64 hash as `flashinfer::sampling::Dist`, so both sides
# sample the same rows. Timed from CUDA-graph replays: 100 launches per graph, one replay
# between two CUDA events, best of 5.
#
# seed and offset are passed explicitly. Otherwise FlashInfer reserves them on the host
# (get_seed_and_offset: a lock plus generator get_state/set_state), which is Python work a
# CUDA graph does not capture, so graph timing would exclude it anyway; the kernel reads
# seed and offset as scalars either way. top_k_top_p uses filter_apply_order="joint" (the
# TopKTopPSamplingFromProbKernel that cute-rs ports); the default "top_k_first" (radix top-k
# then top-p over k entries) is timed too, for reference.
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
SEED, OFFSET = 42, 0
BENCH = [
    ("sampling", 0, 1.0, lambda p, k, pp: S.sampling_from_probs(p, seed=SEED, offset=OFFSET)),
    ("top_k", 10, 1.0, lambda p, k, pp: S.top_k_sampling_from_probs(p, k, seed=SEED, offset=OFFSET)),
    ("top_k", 1000, 1.0, lambda p, k, pp: S.top_k_sampling_from_probs(p, k, seed=SEED, offset=OFFSET)),
    ("top_p", 0, 0.5, lambda p, k, pp: S.top_p_sampling_from_probs(p, pp, seed=SEED, offset=OFFSET)),
    ("top_p", 0, 0.9, lambda p, k, pp: S.top_p_sampling_from_probs(p, pp, seed=SEED, offset=OFFSET)),
    (
        "top_k_top_p",
        50,
        0.9,
        lambda p, k, pp: S.top_k_top_p_sampling_from_probs(
            p, k, pp, filter_apply_order="joint", seed=SEED, offset=OFFSET
        ),
    ),
    (
        "k_first",
        50,
        0.9,
        lambda p, k, pp: S.top_k_top_p_sampling_from_probs(p, k, pp, seed=SEED, offset=OFFSET),
    ),
]

print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
d = 128512
batches = (1, 16, 128, 512)
print("kernel        k     p     distribution  batch  time(us)")
for kind, param, dname in (("normal", 1.0, "normal(1)"), ("gumbel", 0.1, "gumbel(0.1)")):
    allp = probs_of(kind, param, batches[-1], d)
    for name, k, pp, fn in BENCH:
        for b in batches:
            p = allp[:b].contiguous()
            t = graph_bench(lambda: fn(p, k, pp))
            print(f"{name:12}  {k:4}  {pp:.1f}  {dname:12}  {b:5}  {t:8.2f}")
