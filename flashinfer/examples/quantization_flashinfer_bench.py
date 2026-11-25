# FlashInfer 0.7.5 packbits / segment_packbits on the shapes of cute-rs's `quantization`
# example, timed from CUDA-graph replays (100 launches per replay). Run on the pod with
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/workspace/cute-rs-builds/tmp/fi-jit \
#   python fi_bench_quantization.py
import numpy as np
import torch
import flashinfer

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



print("packbits, big bit order")
for log in [16, 20, 24, 26]:
    n = 1 << log
    x = torch.randint(0, 2, (n,), device=dev, dtype=torch.bool)
    y = flashinfer.packbits(x)
    assert (y.cpu().numpy() == np.packbits(x.cpu().numpy())).all()
    t = graph_bench(lambda: flashinfer.packbits(x))
    print(f"2^{log:<9} {t:9.2f}  {(n + n // 8) / (t * 1e-6) / 1e9:8.1f}")

# segment_packbits builds its output indptr with torch ops (and a host sync) around the
# kernel, so it cannot be graph-captured: time the kernel through the JIT module directly.
from flashinfer.quantization.packbits import get_quantization_module

print("segment_packbits kernel, big bit order")
for batch, ln in [(8, 4096), (64, 16385), (512, 4097), (64, 65536)]:
    n = batch * ln
    x = torch.randint(0, 2, (n,), device=dev, dtype=torch.bool)
    indptr = torch.arange(0, (batch + 1) * ln, ln, device=dev, dtype=torch.int32)
    packed = (ln + 7) // 8
    new = torch.arange(0, (batch + 1) * packed, packed, device=dev, dtype=torch.int32)
    y = torch.empty(batch * packed, dtype=torch.uint8, device=dev)
    mod = get_quantization_module()
    mod.segment_packbits(x, indptr, new, "big", y)
    want = np.concatenate([np.packbits(x[i * ln:(i + 1) * ln].cpu().numpy()) for i in range(batch)])
    assert (y.cpu().numpy() == want).all()
    t = graph_bench(lambda: mod.segment_packbits(x, indptr, new, "big", y))
    print(f"{batch:4} x {ln:<8} {t:9.2f}  {(n + n // 8) / (t * 1e-6) / 1e9:8.1f}")
