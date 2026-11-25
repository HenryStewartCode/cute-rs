# FlashInfer (fa2) attention variants on the shapes of cute-rs's `batch_decode` and
# `prefill` examples, GPU time from CUDA-graph replays (as cute-rs's
# `BenchmarkOptions::use_cuda_graphs(true)`):
# * batch decode, page 16, 32 query / 8 KV heads, head dim 128, f16: plain, soft-cap 30,
#   window_left 1024; batch {1, 16, 64} x kv_len {1024, 4096};
# * batch prefill, same heads, qo_len = kv_len: causal, causal + window_left 1024,
#   causal + soft-cap 30, and a causal pattern given as a custom mask (non-causal plan,
#   up to 2048 tokens); batch {1, 8} x {512, 2048, 8192}.
#
# Run on the pod:
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/workspace/cute-rs-builds/tmp/fi-jit \
#     python fi_bench_attention_variants.py
import os
import torch
import flashinfer

HEAD_DIM, QO_HEADS, KV_HEADS, PAGE = 128, 32, 8, 16
dev = "cuda"


def paged(batch, kv_len):
    pages_per = (kv_len + PAGE - 1) // PAGE
    num_pages = batch * pages_per + 2
    k = torch.randn(num_pages, PAGE, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    v = torch.randn_like(k)
    perm = torch.randperm(num_pages, device=dev, dtype=torch.int32)
    indices = perm[: batch * pages_per].contiguous()
    indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages_per
    last = torch.full((batch,), kv_len - (pages_per - 1) * PAGE, device=dev, dtype=torch.int32)
    return k, v, indptr, indices, last


def graph_bench(fn, reps=100):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / reps


torch.manual_seed(0)
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

if not os.environ.get("PREFILL_ONLY"):
    print("decode variant  batch  kv_len  graph(us)")
    for name, kw in [("plain", {}), ("soft-cap 30", {"logits_soft_cap": 30.0}), ("window 1024", {"window_left": 1024})]:
        for batch in (1, 16, 64):
            for kv_len in (1024, 4096):
                k, v, indptr, indices, last = paged(batch, kv_len)
                q = torch.randn(batch, QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
                w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
                w.plan(indptr, indices, last, QO_HEADS, KV_HEADS, HEAD_DIM, PAGE,
                       q_data_type=torch.float16, kv_data_type=torch.float16, **kw)
                out = torch.empty_like(q)
                tg = graph_bench(lambda: w.run(q, (k, v), out=out))
                print(f"{name:14}  {batch:5}  {kv_len:6}  {tg:9.1f}")

print("prefill variant  batch  qo=kv   gpu(us)")
for name, kw in [("causal", {"causal": True}),
                 ("window 1024", {"causal": True, "window_left": 1024}),
                 ("soft-cap 30", {"causal": True, "logits_soft_cap": 30.0}),
                 ("mask", None)]:
    for batch in (1, 8):
        for n in (512, 2048, 8192):
            if kw is None and n > 2048:
                continue
            k, v, indptr, indices, last = paged(batch, n)
            qo_indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * n
            q = torch.randn(batch * n, QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
            args = (qo_indptr, indptr, indices, last, QO_HEADS, KV_HEADS, HEAD_DIM, PAGE)
            if kw is None:
                tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev)).flatten()
                w.plan(*args, custom_mask=tri.repeat(batch), causal=False,
                       q_data_type=torch.float16, kv_data_type=torch.float16)
            else:
                w.plan(*args, q_data_type=torch.float16, kv_data_type=torch.float16, **kw)
            out = torch.empty_like(q)
            lse = torch.empty(q.shape[0], QO_HEADS, dtype=torch.float32, device=dev)
            reps = 100 if batch * n * n <= 8 * 2048 * 2048 else 20
            tg = graph_bench(lambda: w.run(q, (k, v), out=out, lse=lse, return_lse=True), reps)
            print(f"{name:15}  {batch:5}  {n:5}  {tg:9.1f}")
            del q, k, v, out, lse, w
            torch.cuda.empty_cache()
