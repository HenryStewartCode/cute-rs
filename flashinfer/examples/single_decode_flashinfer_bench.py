# FlashInfer single_decode_with_kv_cache (NHD, CUDA cores) on the shapes of cute-rs's
# `single_decode` example: kv_len {1024, 8192, 32768}, 32 query / 8 KV heads, head dim
# 128, f16. GPU time from CUDA-graph replays (100 runs per graph), as cute-rs's
# `BenchmarkOptions::use_cuda_graphs(true)`.
#
# Run on the pod:
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/workspace/cute-rs-builds/tmp/fi-jit \
#     python fi_bench_single_decode.py
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


torch.manual_seed(0)
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
print("kv_len   graph(us)   GB/s (K+V read)")
for n in (1024, 8192, 32768):
    q = torch.randn(QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    k = torch.randn(n, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
    v = torch.randn_like(k)
    o = flashinfer.single_decode_with_kv_cache(q, k, v)
    kk = k.float().repeat_interleave(QO_HEADS // KV_HEADS, dim=1)
    vv = v.float().repeat_interleave(QO_HEADS // KV_HEADS, dim=1)
    ref = torch.einsum("hn,nhd->hd", (torch.einsum("hd,nhd->hn", q.float(), kk) / HEAD_DIM ** 0.5).softmax(-1), vv)
    err = (o.float() - ref).abs().max().item()
    fn = lambda: flashinfer.single_decode_with_kv_cache(q, k, v, return_lse=True)
    tg = graph_bench(fn)
    print(f"{n:6}  {tg:9.1f}  {2 * n * KV_HEADS * HEAD_DIM * 2 / (tg * 1e-6) / 1e9:8.1f}   (|o - ref| {err:.1e})")
