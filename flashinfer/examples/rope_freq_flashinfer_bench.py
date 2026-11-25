# FlashInfer 0.7.5 apply_rope_pos_ids_inplace / apply_llama31_rope_pos_ids_inplace on the
# shapes of cute-rs's `rope` example (f16, head dim 128, heads 32/8), timed from CUDA-graph
# replays. Run on the pod with
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/workspace/cute-rs-builds/tmp/fi-jit \
#   python fi_bench_rope_freq.py
import torch
import flashinfer

dev = "cuda"
HD = 128


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



def row(label, fn_name, rd, inter, nnz, nq=32, nk=8):
    pos = torch.randint(0, 4096, (nnz,), device=dev, dtype=torch.int32)
    q = torch.rand(nnz, nq, HD, device=dev, dtype=torch.float16) * 2 - 1
    k = torch.rand(nnz, nk, HD, device=dev, dtype=torch.float16) * 2 - 1
    fn = getattr(flashinfer, fn_name)
    f = lambda: fn(q, k, pos, rotary_dim=rd, interleave=inter)
    t = graph_bench(f)
    print(f"{label:16} {rd:6} {str(inter):9} {nnz:5}  {t:9.2f}")


print("kind             rotary interleave   nnz  graph(us)")
for nnz in [1, 16, 512, 4096]:
    row("rope", "apply_rope_pos_ids_inplace", 128, False, nnz)
    row("rope", "apply_rope_pos_ids_inplace", 128, True, nnz)
    row("rope rotary 64", "apply_rope_pos_ids_inplace", 64, False, nnz)
    row("llama31", "apply_llama31_rope_pos_ids_inplace", 128, False, nnz)
    row("llama31", "apply_llama31_rope_pos_ids_inplace", 128, True, nnz)
