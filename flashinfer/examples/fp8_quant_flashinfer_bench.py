# FlashInfer 0.7.5 per_token_group_quant_8bit (backend="cuda") on the shapes of cute-rs's
# `quantization_fp8` example, timed from CUDA-graph replays (100 launches per replay). Run on the pod:
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/workspace/cute-rs-builds/tmp/fi-jit \
#   python fi_bench_fp8_quant.py
import torch
from flashinfer.quantization import per_token_group_quant_8bit

dev = "cuda"


def graph_bench(fn, reps=100, trials=5):
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
    best = []
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        best.append(start.elapsed_time(end) * 1000 / reps)
    best.sort()
    return best[len(best) // 2]


def run(x, dst, cm, ue):
    return per_token_group_quant_8bit(
        x, 128, dst_dtype=dst, column_major_scales=cm, scale_ue8m0=ue, backend="cuda"
    )


print("per_token_group_quant_8bit, bf16 -> e4m3, group 128")
print("mode     tokens x hidden   graph(us)  GB/s (in + out)")
for name, cm, ue in [("row", False, False), ("col", True, False), ("ue8m0", True, True)]:
    for hidden in [4096, 8192]:
        for tokens in [1, 16, 128, 1024, 8192]:
            x = torch.randn(tokens, hidden, device=dev, dtype=torch.bfloat16)
            q, s = run(x, torch.float8_e4m3fn, cm, ue)
            ref = (x.float().view(tokens, -1, 128).abs().amax(-1).clamp_min(1e-10) / 448)
            if not ue:
                assert torch.allclose(s.float().reshape(tokens, -1), ref, rtol=1e-6)
            t = graph_bench(lambda: run(x, torch.float8_e4m3fn, cm, ue))
            nbytes = tokens * hidden * 3 + tokens * hidden // 128 * 4
            print(f"{name:<8} {tokens:5} x {hidden:<6}  {t:9.2f}  {nbytes / (t * 1e-6) / 1e9:8.1f}")

print("fp16 -> e5m2 / int8, row scales, 8192 x 4096")
x = torch.randn(8192, 4096, device=dev, dtype=torch.float16)
for name, dst in [("e5m2", torch.float8_e5m2), ("int8", torch.int8)]:
    print(f"{name:<5} {graph_bench(lambda: run(x, dst, False, False)):9.2f} us")
