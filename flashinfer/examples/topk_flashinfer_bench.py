# FlashInfer's `top_k_page_table_transform` and `top_k_ragged_transform` on the shapes of
# cute-rs's `sampling` example (`TRANSFORM_ONLY=1`): benchmarks/bench_topk.py's randn float32
# scores, k = 2048, batch 1/4/16/64, rows of 4096 .. 131072 scores, window lengths either all
# `seq` ("full") or spread over [seq/4, seq] ("var"), page table = arange. Default dispatch
# (radix / CUB, whichever FlashInfer picks; dsa_graph_safe needs 128 KiB of shared memory, absent on SM89), timed from CUDA-graph replays: 100 launches per
# graph, one replay between two CUDA events, best of 5.
import torch
import flashinfer

dev = "cuda"


def graph_bench(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(100):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = None
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        t = a.elapsed_time(b) * 1000 / 100
        best = t if best is None else min(best, t)
    return best


def lengths_of(rows, seq, full):
    if full:
        return torch.full((rows,), seq, dtype=torch.int32, device=dev)
    i = torch.arange(rows, dtype=torch.int64)
    return (seq // 4 + (i * 2654435761) % (3 * seq // 4 + 1)).to(torch.int32).to(dev)


print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
print("kernel                 lengths  batch  seq     k      time(us)")
k = 2048
for seq in [4096, 16384, 65536, 131072]:
    for b in [1, 4, 16, 64]:
        scores = torch.randn(b, seq, device=dev, dtype=torch.float32)
        table = torch.arange(seq, device=dev, dtype=torch.int32).unsqueeze(0).expand(b, -1).contiguous()
        offsets = torch.arange(b, device=dev, dtype=torch.int32) * seq
        for full in [True, False]:
            lengths = lengths_of(b, seq, full)
            tag = "full" if full else "var"
            t = graph_bench(lambda: flashinfer.top_k_page_table_transform(scores, table, lengths, k))
            print(f"{'PageTable':21} {tag:7}  {b:5}  {seq:6} {k:5}  {t:8.2f}", flush=True)
            t = graph_bench(lambda: flashinfer.top_k_ragged_transform(scores, offsets, lengths, k))
            print(f"{'Ragged':21} {tag:7}  {b:5}  {seq:6} {k:5}  {t:8.2f}", flush=True)
