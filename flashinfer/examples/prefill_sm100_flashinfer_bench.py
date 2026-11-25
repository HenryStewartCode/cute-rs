# FlashInfer's Blackwell ragged prefill, BatchPrefillWithRaggedKVCacheWrapper with
# backend="cutlass" (the CUTLASS SM100 FMHA, `fmha_cutlass_sm100.cu`), on the shapes of
# cute-rs's `prefill_sm100` example: batch {1, 8} x qo_len = kv_len {1K, 4K, 16K}, 32 query /
# 8 KV heads, head dim 128, f16 and bf16, causal and not, NHD. GPU time from CUDA-graph
# replays, as cute-rs's `BenchmarkOptions::use_cuda_graphs(true)`; FLOPs as
# flashinfer.testing.attention_flops (causal halves the score matrix). The kernel that runs
# is printed once (torch.profiler), to confirm the backend.
#
# Run on the B200:
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/root/fi-sm100-prefill/fi-jit \
#     python prefill_sm100_flashinfer_bench.py
import torch
import flashinfer

HEAD_DIM, QO_HEADS, KV_HEADS = 128, 32, 8
dev = "cuda"


def graph_bench(fn, reps=30):
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
    pairs = (2 * kv - qo) * qo / 2 if causal else qo * kv
    return batch * pairs * QO_HEADS * 4 * HEAD_DIM


def wrapper(ws, indptr, causal, dtype):
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="cutlass")
    w.plan(indptr, indptr, QO_HEADS, KV_HEADS, HEAD_DIM, causal=causal,
           q_data_type=dtype, kv_data_type=dtype)
    return w


torch.manual_seed(0)
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

# Which kernel runs.
indptr = torch.tensor([0, 1024], device=dev, dtype=torch.int32)
q = torch.randn(1024, QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
k = torch.randn(1024, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
w = wrapper(ws, indptr, True, torch.float16)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    w.run(q, k, k)
    torch.cuda.synchronize()
names = {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}
print("kernels:", sorted(n[:90] for n in names))

print("dtype causal batch  seq   gpu(us)  TFLOPS")
for dtype in (torch.float16, torch.bfloat16):
    for causal in (False, True):
        for batch in (1, 8):
            for n in (1024, 4096, 16384):
                indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * n
                q = torch.randn(batch * n, QO_HEADS, HEAD_DIM, dtype=dtype, device=dev)
                k = torch.randn(batch * n, KV_HEADS, HEAD_DIM, dtype=dtype, device=dev)
                v = torch.randn_like(k)
                w = wrapper(ws, indptr, causal, dtype)
                out = torch.empty_like(q)
                lse = torch.empty(q.shape[0], QO_HEADS, dtype=torch.float32, device=dev)
                fn = lambda: w.run(q, k, v, out=out, lse=lse, return_lse=True)
                tg = graph_bench(fn)
                name = "f16" if dtype == torch.float16 else "bf16"
                print(f"{name:5} {str(causal):6} {batch:5} {n:5} {tg:9.1f} {flops(batch, n, n, causal) / (tg * 1e-6) / 1e12:7.1f}")
                del q, k, v, out, lse, w
