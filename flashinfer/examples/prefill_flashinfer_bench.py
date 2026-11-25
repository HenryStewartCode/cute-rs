# FlashInfer batch prefill over a paged KV cache (BatchPrefillWithPagedKVCacheWrapper,
# backend "fa2") on the shapes of cute-rs's `prefill` example: batch {1, 8} x
# qo_len = kv_len {512, 2048, 8192}, causal, page size 16, 32 query / 8 KV heads, head dim
# 128, f16, NHD. GPU time from CUDA-graph replays (100 runs per graph), as cute-rs's
# `BenchmarkOptions::use_cuda_graphs(true)`. FLOPs as flashinfer.testing.attention_flops.
#
# Run on the pod:
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/workspace/cute-rs-builds/tmp/fi-jit \
#     python fi_bench_prefill.py
import torch
import flashinfer

HEAD_DIM, QO_HEADS, KV_HEADS, PAGE = 128, 32, 8, 16
dev = "cuda"


def problem(batch, qo_len, kv_len):
    pages_per = (kv_len + PAGE - 1) // PAGE
    num_pages = batch * pages_per + 2
    k = torch.randn(num_pages, PAGE, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    v = torch.randn_like(k)
    perm = torch.randperm(num_pages, device=dev, dtype=torch.int32)
    indices = perm[: batch * pages_per].contiguous()
    kv_indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages_per
    last = torch.full((batch,), kv_len - (pages_per - 1) * PAGE, device=dev, dtype=torch.int32)
    qo_indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * qo_len
    q = torch.randn(batch * qo_len, QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    return q, k, v, qo_indptr, kv_indptr, indices, last


def wrapper(qo_indptr, kv_indptr, indices, last, causal):
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    w.plan(qo_indptr, kv_indptr, indices, last, QO_HEADS, KV_HEADS, HEAD_DIM, PAGE,
           causal=causal, q_data_type=torch.float16, kv_data_type=torch.float16)
    return w


def reference(q, k, v, qo_indptr, kv_indptr, indices, last, b, causal):
    pages = indices[kv_indptr[b]:kv_indptr[b + 1]].long()
    n = (len(pages) - 1) * PAGE + int(last[b])
    kk = k[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:n].float().repeat_interleave(QO_HEADS // KV_HEADS, dim=1)
    vv = v[pages].reshape(-1, KV_HEADS, HEAD_DIM)[:n].float().repeat_interleave(QO_HEADS // KV_HEADS, dim=1)
    qq = q[qo_indptr[b]:qo_indptr[b + 1]].float()
    s = torch.einsum("qhd,nhd->hqn", qq, kk) / HEAD_DIM ** 0.5
    if causal:
        lq = qq.shape[0]
        mask = torch.arange(n, device=dev)[None, :] > (torch.arange(lq, device=dev)[:, None] + n - lq)
        s = s.masked_fill(mask[None], float("-inf"))
    return torch.einsum("hqn,nhd->qhd", s.softmax(-1), vv)


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


def flops(batch, qo, kv, causal):
    if causal:
        return batch * (2 * kv - qo) * qo * QO_HEADS * 2 * HEAD_DIM
    return 4 * batch * qo * kv * QO_HEADS * HEAD_DIM


torch.manual_seed(0)
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
for causal in (False, True):
    q, k, v, qo_indptr, kv_indptr, indices, last = problem(3, 77, 300)
    w = wrapper(qo_indptr, kv_indptr, indices, last, causal)
    o = w.run(q, (k, v))
    err = max((o[qo_indptr[b]:qo_indptr[b + 1]].float()
               - reference(q, k, v, qo_indptr, kv_indptr, indices, last, b, causal)).abs().max().item()
              for b in range(3))
    print(f"flashinfer vs torch (causal {causal}): max |o - ref| = {err:.2e}")

print("batch  qo=kv  causal   gpu(us)   TFLOPS")
for batch in (1, 8):
    for n in (512, 2048, 8192):
        q, k, v, qo_indptr, kv_indptr, indices, last = problem(batch, n, n)
        w = wrapper(qo_indptr, kv_indptr, indices, last, True)
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[0], QO_HEADS, dtype=torch.float32, device=dev)
        fn = lambda: w.run(q, (k, v), out=out, lse=lse, return_lse=True)
        reps = 100 if batch * n * n <= 8 * 2048 * 2048 else 20
        tg = graph_bench(fn, reps)
        print(f"{batch:5}  {n:5}  {True!s:6}  {tg:9.1f}  {flops(batch, n, n, True) / (tg * 1e-6) / 1e12:7.2f}")
        del q, k, v, out, lse, w
        torch.cuda.empty_cache()
