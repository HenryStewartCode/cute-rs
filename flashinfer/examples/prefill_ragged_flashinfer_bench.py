# FlashInfer ragged-KV batch prefill (BatchPrefillWithRaggedKVCacheWrapper, backend "fa2")
# and single_prefill_with_kv_cache (backend "fa2") on the shapes of cute-rs's
# `prefill_ragged` example: batch {1, 8} x qo_len = kv_len {512, 2048, 8192}, and one
# request of qo_len = kv_len {1024, 8192, 32768}; causal, 32 query / 8 KV heads, head dim
# 128, f16, NHD. GPU time from CUDA-graph replays, as cute-rs's
# `BenchmarkOptions::use_cuda_graphs(true)`. FLOPs as flashinfer.testing.attention_flops.
#
# Run on the pod:
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/workspace/cute-rs-builds/tmp/fi-jit \
#     python fi_bench_prefill_ragged.py
import torch
import flashinfer

HEAD_DIM, QO_HEADS, KV_HEADS = 128, 32, 8
dev = "cuda"


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


def flops(batch, qo, kv):
    return batch * (2 * kv - qo) * qo * QO_HEADS * 2 * HEAD_DIM


def reference(q, k, v, causal):
    kk = k.float().repeat_interleave(QO_HEADS // KV_HEADS, dim=1)
    vv = v.float().repeat_interleave(QO_HEADS // KV_HEADS, dim=1)
    s = torch.einsum("qhd,nhd->hqn", q.float(), kk) / HEAD_DIM ** 0.5
    if causal:
        lq, n = q.shape[0], k.shape[0]
        mask = torch.arange(n, device=dev)[None, :] > (torch.arange(lq, device=dev)[:, None] + n - lq)
        s = s.masked_fill(mask[None], float("-inf"))
    return torch.einsum("hqn,nhd->qhd", s.softmax(-1), vv)


torch.manual_seed(0)
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
for causal in (False, True):
    q = torch.randn(77, QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    k = torch.randn(300, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    v = torch.randn_like(k)
    o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=causal, backend="fa2")
    print(f"single vs torch (causal {causal}): max |o - ref| = {(o.float() - reference(q, k, v, causal)).abs().max().item():.2e}")

print("api     batch  qo=kv   gpu(us)   TFLOPS")
ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
for batch in (1, 8):
    for n in (512, 2048, 8192):
        indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * n
        q = torch.randn(batch * n, QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
        k = torch.randn(batch * n, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
        v = torch.randn_like(k)
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
        w.plan(indptr, indptr, QO_HEADS, KV_HEADS, HEAD_DIM, causal=True,
               q_data_type=torch.float16, kv_data_type=torch.float16)
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[0], QO_HEADS, dtype=torch.float32, device=dev)
        fn = lambda: w.run(q, k, v, out=out, lse=lse, return_lse=True)
        reps = 100 if batch * n * n <= 8 * 2048 * 2048 else 20
        tg = graph_bench(fn, reps)
        print(f"ragged  {batch:5}  {n:5}  {tg:9.1f}  {flops(batch, n, n) / (tg * 1e-6) / 1e12:7.2f}")
        del q, k, v, out, lse, w
        torch.cuda.empty_cache()
for n, reps in ((1024, 100), (8192, 20), (32768, 5)):
    q = torch.randn(n, QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    k = torch.randn(n, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    v = torch.randn_like(k)
    fn = lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="fa2", return_lse=True)
    tg = graph_bench(fn, reps)
    print(f"single  {1:5}  {n:5}  {tg:9.1f}  {flops(1, n, n) / (tg * 1e-6) / 1e12:7.2f}")
    del q, k, v
    torch.cuda.empty_cache()
