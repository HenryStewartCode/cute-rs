# FlashInfer 0.7.5 paged-KV append into an FP8 cache (FP8 inputs, as FlashInfer requires:
# one element type for inputs and cache) on the append shapes of cute-rs's `rope` example
# (`KV=e4m3`), timed from CUDA-graph replays. Usage: python ... [e4m3|e5m2]
import sys

import torch
import flashinfer

dev = "cuda"
HD = 128
KV = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[sys.argv[1] if len(sys.argv) > 1 else "e4m3"]


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


def append_row(app, label, page=16, heads=8):
    nnz = sum(app)
    seq = app  # the cache holds exactly the appended tokens
    pages_per = [(l + page - 1) // page for l in seq]
    num_pages = sum(pages_per) + 2
    cache = torch.rand(num_pages, 2, page, heads, HD, dtype=torch.float16, device=dev).to(KV)
    k = torch.rand(nnz, heads, HD, dtype=torch.float16, device=dev).to(KV)
    v = torch.rand(nnz, heads, HD, dtype=torch.float16, device=dev).to(KV)
    perm = torch.randperm(num_pages, device=dev, dtype=torch.int32)
    indices = perm[: sum(pages_per)].contiguous()
    indptr = torch.tensor([0] + list(torch.tensor(pages_per).cumsum(0)), device=dev, dtype=torch.int32)
    last = torch.tensor([(l - 1) % page + 1 for l in seq], device=dev, dtype=torch.int32)
    x_indptr = torch.tensor([0] + list(torch.tensor(app).cumsum(0)), device=dev, dtype=torch.int32)
    bi, po = flashinfer.get_batch_indices_positions(x_indptr, flashinfer.get_seq_lens(indptr, last, page), nnz)
    fn = lambda: flashinfer.append_paged_kv_cache(k, v, bi, po, cache, indices, indptr, last, "NHD")
    fn()
    # spot check token 0
    b0, p0 = int(bi[0]), int(po[0])
    pg = int(indices[int(indptr[b0]) + p0 // page])
    ok = torch.equal(cache[pg, 0, p0 % page].view(torch.uint8), k[0].view(torch.uint8)) and torch.equal(cache[pg, 1, p0 % page].view(torch.uint8), v[0].view(torch.uint8))
    t = graph_bench(fn)
    byt = nnz * heads * HD * 1 * 2 * 2
    print(f"{len(app):5} {nnz:6}  {t:9.2f}  {byt / (t * 1e-6) / 1e9:8.1f}   ({label}) ok={ok}")


@torch.inference_mode()
def main():
    print(torch.cuda.get_device_name(), KV)
    print("Append to the paged cache, 8 KV heads, head dim 128, page 16")
    print("batch  tokens  graph(us)  GB/s")
    for n in [1, 16, 512, 4096]:
        append_row([n], f"{n} tokens, 1 request")
    append_row([1] * 8, "decode x8")
    append_row([5000 - 7] + [1] * 7, "prefill + 7 decode")
    append_row([625] * 8, "8 x 625")


main()
