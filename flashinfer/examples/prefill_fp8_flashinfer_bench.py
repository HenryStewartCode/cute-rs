# FlashInfer batch prefill with an FP8 KV cache (BatchPrefillWithPagedKVCacheWrapper,
# backend "fa2", float16 queries, kv_data_type float8_e4m3fn / float8_e5m2, k_scale 0.5 and
# v_scale 1.5) on the shapes of cute-rs's `prefill` example (FP8 part): batch {1, 8} x
# qo_len = kv_len {512, 2048, 8192}, causal, page size 16, 32 query / 8 KV heads, head dim
# 128, NHD. Float16 KV is timed too, for the FP8 speed-up. GPU time from CUDA-graph replays
# (100 runs per graph; 20 for the largest shapes), as cute-rs's
# `BenchmarkOptions::use_cuda_graphs(true)`. FLOPs as flashinfer.testing.attention_flops.
#
# Run on the pod:
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/workspace/cute-rs-builds/tmp/fi-jit \
#     python fi_bench_prefill_fp8.py
import torch
import flashinfer

HEAD_DIM, QO_HEADS, KV_HEADS, PAGE = 128, 32, 8, 16
K_SCALE, V_SCALE = 0.5, 1.5
dev = "cuda"
DTYPES = {"F16": torch.float16, "E4M3": torch.float8_e4m3fn, "E5M2": torch.float8_e5m2}


def problem(batch, qo_len, kv_len, kv_dtype):
    pages_per = (kv_len + PAGE - 1) // PAGE
    num_pages = batch * pages_per + 2
    k = torch.randn(num_pages, PAGE, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    v = torch.randn_like(k)
    if kv_dtype != torch.float16:
        k, v = (k / K_SCALE).to(kv_dtype), (v / V_SCALE).to(kv_dtype)
    perm = torch.randperm(num_pages, device=dev, dtype=torch.int32)
    indices = perm[: batch * pages_per].contiguous()
    kv_indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages_per
    last = torch.full((batch,), kv_len - (pages_per - 1) * PAGE, device=dev, dtype=torch.int32)
    qo_indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * qo_len
    q = torch.randn(batch * qo_len, QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    return q, k, v, qo_indptr, kv_indptr, indices, last


def wrapper(qo_indptr, kv_indptr, indices, last, causal, kv_dtype):
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    w.plan(qo_indptr, kv_indptr, indices, last, QO_HEADS, KV_HEADS, HEAD_DIM, PAGE,
           causal=causal, q_data_type=torch.float16, kv_data_type=kv_dtype)
    return w


def run_args(kv_dtype):
    return {} if kv_dtype == torch.float16 else dict(k_scale=K_SCALE, v_scale=V_SCALE)


def reference(q, k, v, qo_indptr, kv_indptr, indices, last, b, causal, kv_dtype):
    sk, sv = (1.0, 1.0) if kv_dtype == torch.float16 else (K_SCALE, V_SCALE)
    pages = indices[kv_indptr[b]:kv_indptr[b + 1]].long()
    n = (len(pages) - 1) * PAGE + int(last[b])
    kk = (k[pages].float() * sk).reshape(-1, KV_HEADS, HEAD_DIM)[:n].repeat_interleave(QO_HEADS // KV_HEADS, dim=1)
    vv = (v[pages].float() * sv).reshape(-1, KV_HEADS, HEAD_DIM)[:n].repeat_interleave(QO_HEADS // KV_HEADS, dim=1)
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
for name, dt in DTYPES.items():
    for causal in (False, True):
        q, k, v, qo_indptr, kv_indptr, indices, last = problem(3, 77, 300, dt)
        w = wrapper(qo_indptr, kv_indptr, indices, last, causal, dt)
        o = w.run(q, (k, v), **run_args(dt))
        err = max((o[qo_indptr[b]:qo_indptr[b + 1]].float()
                   - reference(q, k, v, qo_indptr, kv_indptr, indices, last, b, causal, dt)).abs().max().item()
                  for b in range(3))
        print(f"flashinfer {name} vs torch (causal {causal}): max |o - ref| = {err:.2e}")

print("kv      batch  qo=kv  causal   gpu(us)   TFLOPS")
for batch in (1, 8):
    for n in (512, 2048, 8192):
        for name, dt in DTYPES.items():
            q, k, v, qo_indptr, kv_indptr, indices, last = problem(batch, n, n, dt)
            w = wrapper(qo_indptr, kv_indptr, indices, last, True, dt)
            out = torch.empty_like(q)
            lse = torch.empty(q.shape[0], QO_HEADS, dtype=torch.float32, device=dev)
            fn = lambda: w.run(q, (k, v), out=out, lse=lse, return_lse=True, **run_args(dt))
            reps = 100 if batch * n * n <= 8 * 2048 * 2048 else 20
            tg = graph_bench(fn, reps)
            print(f"{name:6}  {batch:5}  {n:5}  {True!s:6}  {tg:9.1f}  {flops(batch, n, n, True) / (tg * 1e-6) / 1e12:7.2f}")
            del q, k, v, out, lse, w
            torch.cuda.empty_cache()
