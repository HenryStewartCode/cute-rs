# FlashInfer 0.7.5's SegmentGEMMWrapper.run (backend "auto" -> sm80 CUTLASS grouped GEMM on SM89)
# and bmm_fp8 (cuBLASLt on SM89) on the shapes of cute-rs's flashinfer gemm example, GPU time
# from CUDA-graph replays (100 launches captured, one replay timed, average per launch).
import os
import torch
import flashinfer

dev = "cuda"


def graph_bench(fn, n=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / n


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    amax = x.abs().max().clamp(min=1e-12)
    scale = finfo.max / amax
    return (x * scale).clamp(min=finfo.min, max=finfo.max).to(dtype), scale.float().reciprocal()


ws = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=dev)
seg = flashinfer.gemm.SegmentGEMMWrapper(ws, backend="auto")

# Correctness of FlashInfer itself on one shape (and its accumulate-into-out semantics).
x = torch.randn(37, 256, device=dev).bfloat16()
w = torch.randn(2, 384, 256, device=dev).bfloat16()
indptr = torch.tensor([0, 7, 37], dtype=torch.int64, device=dev)
y = torch.ones(37, 384, dtype=torch.bfloat16, device=dev)
seg.run(x, w, 2, True, out=y, seg_indptr=indptr)
ref = torch.cat([x[:7].float() @ w[0].float().T, x[7:].float() @ w[1].float().T]) + 1
print("segment check: max abs err %.3f (accumulates into out: %s)" % ((y.float() - ref).abs().max().item(),
      (y.float() - ref).abs().max().item() < 1.0))

ds = [4096, 8192] if os.environ.get("D8192") else [4096]
print("segment_gemm bf16 col-major  batch tokens  d_in  d_out  time(us)  TFLOPS")
for d in ds:
    for batch in [1, 3, 8, 16]:
        for tokens in [32, 128, 512]:
            W = torch.randn(batch, d, d, device=dev).bfloat16()
            X = torch.randn(batch * tokens, d, device=dev).bfloat16()
            Y = torch.empty(batch * tokens, d, dtype=torch.bfloat16, device=dev)
            ip = torch.arange(0, (batch + 1) * tokens, tokens, dtype=torch.int64, device=dev)
            t = graph_bench(lambda: seg.run(X, W, batch, True, out=Y, seg_indptr=ip))
            print("%28s %5d %6d %5d %6d %9.1f %7.2f" % ("", batch, tokens, d, d, t, 2 * batch * tokens * d * d / t * 1e-6))
            del W, X, Y

print("bmm_fp8 e4m3 -> bf16   batch     m     n     k  time(us)  TFLOPS")
for b, m, n, k in [(64, 4, 1024, 7168), (1, 64, 4096, 4096)]:
    A, As = to_float8(torch.randn(b, m, k, device=dev))
    B, Bs = to_float8(torch.randn(b, n, k, device=dev))
    B = B.transpose(-2, -1)
    out = torch.empty(b, m, n, dtype=torch.bfloat16, device=dev)
    flashinfer.bmm_fp8(A, B, As, Bs, torch.bfloat16, out=out)
    ref = torch.bmm(A.float(), B.float()) * As * Bs
    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    t = graph_bench(lambda: flashinfer.bmm_fp8(A, B, As, Bs, torch.bfloat16, out=out))
    print("%22s %5d %5d %5d %5d %9.1f %7.2f   (rel err %.1e)" % ("", b, m, n, k, t, 2 * b * m * n * k / t * 1e-6, err))
