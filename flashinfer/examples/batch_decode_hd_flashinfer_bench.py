# FlashInfer batch decode (fa2) at head dimension 64/128/256 and f16/bf16 on the shapes of
# cute-rs's batch_decode example (`HD=..`, `DT=bf16`), timed from CUDA-graph replays like
# cute::testing::benchmark.
# Usage: python batch_decode_hd_flashinfer_bench.py [head_dim] [f16|bf16]
import sys

import torch
import flashinfer

HEAD_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 128
DT = torch.bfloat16 if (sys.argv[2] if len(sys.argv) > 2 else "f16") == "bf16" else torch.float16
GROUP, KV_HEADS, PAGE = 4, 8, 16
dev = "cuda"


def problem(batch, kv_len):
    pages_per = (kv_len + PAGE - 1) // PAGE
    num_pages = batch * pages_per + 2
    k = torch.randn(num_pages, PAGE, KV_HEADS, HEAD_DIM, dtype=DT, device=dev)
    v = torch.randn_like(k)
    perm = torch.randperm(num_pages, device=dev, dtype=torch.int32)
    indices = perm[: batch * pages_per].contiguous()
    indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages_per
    last = torch.full((batch,), kv_len - (pages_per - 1) * PAGE, device=dev, dtype=torch.int32)
    q = torch.randn(batch, KV_HEADS * GROUP, HEAD_DIM, dtype=DT, device=dev)
    return q, k, v, indptr, indices, last


def wrapper(indptr, indices, last):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=False, backend="fa2")
    w.plan(indptr, indices, last, KV_HEADS * GROUP, KV_HEADS, HEAD_DIM, PAGE, q_data_type=DT, kv_data_type=DT)
    return w


def reference(q, k, v, indptr, indices, last, b):
    pages = indices[indptr[b]:indptr[b + 1]].long()
    n = (len(pages) - 1) * PAGE + int(last[b])
    kk = k[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:n].double().repeat_interleave(GROUP, dim=1)
    vv = v[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:n].double().repeat_interleave(GROUP, dim=1)
    s = torch.einsum("hd,nhd->hn", q[b].double(), kk) / HEAD_DIM ** 0.5
    return torch.einsum("hn,nhd->hd", s.softmax(-1), vv)


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
    best = float("inf")
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / 100)
    return best


torch.manual_seed(0)
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name(), "head dim", HEAD_DIM, DT)
q, k, v, indptr, indices, last = problem(8, 1000)
w = wrapper(indptr, indices, last)
o = w.run(q, (k, v))
err = max((o[b].double() - reference(q, k, v, indptr, indices, last, b)).abs().max().item() for b in range(8))
print(f"flashinfer vs torch: max |o - ref| = {err:.2e}")

print("batch  kv_len  graph(us)   GB/s (K+V read)")
for batch in (1, 16, 64):
    for kv_len in (1024, 4096):
        q, k, v, indptr, indices, last = problem(batch, kv_len)
        w = wrapper(indptr, indices, last)
        out = torch.empty_like(q)
        fn = lambda: w.run(q, (k, v), out=out)
        tg = graph_bench(fn)
        gbs = 2 * batch * kv_len * KV_HEADS * HEAD_DIM * 2 / (tg * 1e-6) / 1e9
        print(f"{batch:5}  {kv_len:6}  {tg:9.1f}  {gbs:8.1f}")
