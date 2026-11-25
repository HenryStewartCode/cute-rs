# FlashInfer batch decode (fa2) with an FP8 K/V cache (kv_data_type=float8_e4m3fn or
# float8_e5m2, f16 queries and output) on the shapes of cute-rs's batch_decode example
# (`KV=e4m3`), timed from CUDA-graph replays like cute::testing::benchmark.
# Usage: python batch_decode_fp8_flashinfer_bench.py [e4m3|e5m2]
import sys

import torch
import flashinfer

HEAD_DIM, GROUP, KV_HEADS, PAGE = 128, 4, 8, 16
dev = "cuda"
KV = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[sys.argv[1] if len(sys.argv) > 1 else "e4m3"]


def problem(batch, kv_len):
    pages_per = (kv_len + PAGE - 1) // PAGE
    num_pages = batch * pages_per + 2
    k = torch.randn(num_pages, PAGE, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev).to(KV)
    v = torch.randn(num_pages, PAGE, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev).to(KV)
    perm = torch.randperm(num_pages, device=dev, dtype=torch.int32)
    indices = perm[: batch * pages_per].contiguous()
    indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages_per
    last = torch.full((batch,), kv_len - (pages_per - 1) * PAGE, device=dev, dtype=torch.int32)
    q = torch.randn(batch, KV_HEADS * GROUP, HEAD_DIM, dtype=torch.float16, device=dev)
    return q, k, v, indptr, indices, last


def wrapper(indptr, indices, last):
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=False, backend="fa2")
    w.plan(indptr, indices, last, KV_HEADS * GROUP, KV_HEADS, HEAD_DIM, PAGE,
           q_data_type=torch.float16, kv_data_type=KV)
    return w


def reference(q, k, v, indptr, indices, last, b, k_scale=1.0, v_scale=1.0):
    pages = indices[indptr[b]:indptr[b + 1]].long()
    n = (len(pages) - 1) * PAGE + int(last[b])
    kk = k[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:n].double() * k_scale
    vv = v[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:n].double() * v_scale
    kk = kk.repeat_interleave(GROUP, dim=1)
    vv = vv.repeat_interleave(GROUP, dim=1)
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
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name(), KV)
q, k, v, indptr, indices, last = problem(8, 1000)
w = wrapper(indptr, indices, last)
o = w.run(q, (k, v), k_scale=0.5, v_scale=2.0)
err = max((o[b].double() - reference(q, k, v, indptr, indices, last, b, 0.5, 2.0)).abs().max().item()
          for b in range(8))
print(f"flashinfer vs torch (k_scale 0.5, v_scale 2): max |o - ref| = {err:.2e}")

print("batch  kv_len  graph(us)   GB/s (K+V read)")
for batch in (1, 16, 64):
    for kv_len in (1024, 4096):
        q, k, v, indptr, indices, last = problem(batch, kv_len)
        w = wrapper(indptr, indices, last)
        out = torch.empty_like(q)
        fn = lambda: w.run(q, (k, v), out=out)
        tg = graph_bench(fn)
        gbs = 2 * batch * kv_len * KV_HEADS * HEAD_DIM * 1 / (tg * 1e-6) / 1e9
        print(f"{batch:5}  {kv_len:6}  {tg:9.1f}  {gbs:8.1f}")
