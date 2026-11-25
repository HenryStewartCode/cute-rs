# FlashInfer 0.7.5 RoPE-with-cos/sin-cache and paged-KV append on the shapes of cute-rs's
# `rope` example, timed from CUDA-graph replays (100 launches per replay), as cute-rs's
# `BenchmarkOptions::use_cuda_graphs` does. Run on the pod with
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/workspace/cute-rs-builds/tmp/fi-jit \
#   python fi_bench_rope.py
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


def cache_for(max_pos, hd):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float, device=dev) / hd))
    f = torch.einsum("i,j->ij", torch.arange(max_pos, dtype=torch.float, device=dev), inv)
    return torch.cat((f.cos(), f.sin()), dim=-1).contiguous()


def rope_ref(x, cache, pos, hd, neox):
    n = x.shape[0]
    x = x.view(n, -1, hd).float()
    c = cache[pos.long()][:, : hd // 2].unsqueeze(1)
    s = cache[pos.long()][:, hd // 2:].unsqueeze(1)
    if neox:
        a, b = x[..., : hd // 2], x[..., hd // 2:]
        return torch.cat((a * c - b * s, b * c + a * s), -1)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * c - b * s, b * c + a * s), -1).flatten(-2)


def rope_row(dtype, neox, nnz, nq, nk):
    cache = cache_for(max(4096, nnz), HD)
    pos = torch.randint(0, cache.shape[0], (nnz,), device=dev, dtype=torch.int32)
    q = torch.rand(nnz, nq * HD, device=dev, dtype=dtype) * 2 - 1
    k = torch.rand(nnz, nk * HD, device=dev, dtype=dtype) * 2 - 1
    want = rope_ref(q, cache, pos, HD, neox)
    flashinfer.apply_rope_with_cos_sin_cache_inplace(pos, q, k, HD, cache, neox)
    err = (q.view(nnz, -1, HD).float() - want).abs().max().item()
    t = graph_bench(lambda: flashinfer.apply_rope_with_cos_sin_cache_inplace(pos, q, k, HD, cache, neox))
    byt = 2 * nnz * (nq + nk) * HD * 2
    byc = byt + nnz * HD * 4
    name = "f16" if dtype == torch.float16 else "bf16"
    print(f"{name:5} {'neox' if neox else 'gptj':5} {nnz:5} {nq:4} {nk:3}  {t:9.2f}  {byt / (t * 1e-6) / 1e9:8.1f}  {byc / (t * 1e-6) / 1e9:8.1f}  (err {err:.1e})")


def append_row(app, label, page=16, heads=8):
    nnz = sum(app)
    seq = app  # the cache holds exactly the appended tokens
    pages_per = [(l + page - 1) // page for l in seq]
    num_pages = sum(pages_per) + 2
    cache = torch.rand(num_pages, 2, page, heads, HD, dtype=torch.float16, device=dev)
    k = torch.rand(nnz, heads, HD, dtype=torch.float16, device=dev)
    v = torch.rand(nnz, heads, HD, dtype=torch.float16, device=dev)
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
    ok = torch.equal(cache[pg, 0, p0 % page], k[0]) and torch.equal(cache[pg, 1, p0 % page], v[0])
    t = graph_bench(fn)
    byt = nnz * heads * HD * 2 * 2 * 2
    print(f"{len(app):5} {nnz:6}  {t:9.2f}  {byt / (t * 1e-6) / 1e9:8.1f}   ({label}) ok={ok}")


@torch.inference_mode()
def main():
    print(torch.cuda.get_device_name())
    print("RoPE in place, heads nq/nk, head dim 128 (graph replays)")
    print("dtype layout   nnz   nq  nk  graph(us)  GB/s(q,k)  GB/s(+cache)")
    for nnz in [1, 16, 512, 4096]:
        rope_row(torch.float16, True, nnz, 32, 8)
        rope_row(torch.float16, False, nnz, 32, 8)
        rope_row(torch.bfloat16, True, nnz, 32, 8)
    print("bench_rope.py shape: heads 32/32")
    for nnz in [256, 2048, 8192]:
        rope_row(torch.float16, True, nnz, 32, 32)
    print("\nAppend to the paged cache, 8 KV heads, head dim 128, f16, page 16")
    print("batch  tokens  graph(us)  GB/s")
    for n in [1, 16, 512, 4096]:
        append_row([n], f"{n} tokens, 1 request")
    append_row([1] * 8, "decode x8")
    append_row([5000 - 7] + [1] * 7, "prefill + 7 decode")
    append_row([625] * 8, "8 x 625")


main()
