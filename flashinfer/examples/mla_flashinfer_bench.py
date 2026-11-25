# FlashInfer MLA paged attention (BatchMLAPagedAttentionWrapper, backend "fa2", which is
# what backend="auto" picks on SM89) on the shapes of FlashInfer's
# benchmarks/bench_deepseek_mla.py, as cute-rs's `mla` example: decode (one query per
# request), page size 1, head dims 512 + 64, f16, seq_len {1024, 2048, 8192} x batch
# {64, 128, 768} x heads {64, 128} (768 x 8192 skipped: cute-rs's views address at most
# 2^30 cache elements), plus heads 16 at batch 64. GPU time from CUDA-graph replays, as
# cute-rs's `BenchmarkOptions::use_cuda_graphs(true)`. GB/s and TFLOPS as the benchmark
# computes them.
#
# Run on the pod:
#   PATH=/workspace/fi-venv/bin:$PATH FLASHINFER_WORKSPACE_BASE=/workspace/cute-rs-builds/tmp/fi-jit \
#     python fi_bench_mla.py
import torch
import flashinfer

CKV, KPE = 512, 64
dev = "cuda"
ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=dev)


def problem(batch, seq_len, heads):
    q_nope = torch.randn(batch, heads, CKV, dtype=torch.half, device=dev)
    q_pe = torch.randn(batch, heads, KPE, dtype=torch.half, device=dev)
    ckv = torch.randn(batch * seq_len, 1, CKV, dtype=torch.half, device=dev)
    kpe = torch.randn(batch * seq_len, 1, KPE, dtype=torch.half, device=dev)
    q_indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32)
    kv_indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * seq_len
    kv_indices = torch.arange(0, batch * seq_len, device=dev, dtype=torch.int32)
    kv_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=dev)
    return q_nope, q_pe, ckv, kpe, q_indptr, kv_indptr, kv_indices, kv_lens


def wrapper(backend, q_indptr, kv_indptr, kv_indices, kv_lens, heads, causal=False):
    w = flashinfer.mla.BatchMLAPagedAttentionWrapper(ws, backend=backend)
    w.plan(q_indptr, kv_indptr, kv_indices, kv_lens, heads, CKV, KPE, 1, causal,
           1.0 / (CKV + KPE) ** 0.5, torch.half, torch.half)
    return w


def graph_bench(fn, reps):
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
# Which backend "auto" resolves to here, and a correctness check of "fa2" against torch.
q_nope, q_pe, ckv, kpe, q_indptr, kv_indptr, kv_indices, kv_lens = problem(4, 300, 128)
auto = wrapper("auto", q_indptr, kv_indptr, kv_indices, kv_lens, 128)
print("auto backend:", getattr(auto, "_backend", None), type(getattr(auto, "_planned", None)).__name__)
w = wrapper("fa2", q_indptr, kv_indptr, kv_indices, kv_lens, 128)
o = w.run(q_nope, q_pe, ckv, kpe)
k = torch.cat([ckv, kpe], -1).float().reshape(4, 300, CKV + KPE)
q = torch.cat([q_nope, q_pe], -1).float()
s = torch.einsum("bhd,bnd->bhn", q, k) / (CKV + KPE) ** 0.5
ref = torch.einsum("bhn,bnd->bhd", s.softmax(-1), k[..., :CKV])
print(f"fa2 vs torch: max |o - ref| = {(o.float() - ref).abs().max().item():.2e}")

print("batch  seq_len  heads   gpu(us)    GB/s  TFLOPS")
shapes = [(b, n, h) for n in (1024, 2048, 8192) for b in (64, 128, 768) for h in (64, 128)
          if b * n * CKV < 1 << 30] + [(64, 1024, 16), (64, 8192, 16)]
for batch, seq_len, heads in shapes:
    q_nope, q_pe, ckv, kpe, q_indptr, kv_indptr, kv_indices, kv_lens = problem(batch, seq_len, heads)
    w = wrapper("fa2", q_indptr, kv_indptr, kv_indices, kv_lens, heads)
    out = torch.empty_like(q_nope)
    fn = lambda: w.run(q_nope, q_pe, ckv, kpe, out=out)
    reps = 50 if batch * seq_len <= 128 * 2048 else 10
    t = graph_bench(fn, reps)
    io = sum(x.numel() * x.element_size() for x in (q_nope, q_pe, ckv, kpe, out))
    flops = 2 * batch * heads * (2 * CKV + KPE) * seq_len
    print(f"{batch:5}  {seq_len:7}  {heads:5}  {t:8.1f}  {io / (t * 1e-6) / 1e9:6.1f}  {flops / (t * 1e-6) / 1e12:6.2f}")
    del q_nope, q_pe, ckv, kpe, out, w
    torch.cuda.empty_cache()
