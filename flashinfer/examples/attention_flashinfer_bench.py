# FlashInfer's schedulers on the mixes of its benchmarks/bench_batch_attention.py, in f16
# (cute-rs's `attention` example's mixes): decodes of one query next to causal prefill
# chunks, Qwen-8B (32/8) and Llama-3.1-70B (64/8) heads, head dim 128, `arange` page tables,
# page size PAGE (default 1, the benchmark's). Times (microseconds) are CUDA-graph replays:
# BatchPrefillWithPagedKVCache over
# the whole batch, and the benchmark's "Decode + Prefill" (tensor-core batch decode plus
# batch prefill, timed apart and summed). POD (one prefill request) is timed with CUDA events:
# its graph replay skips the scheduler-counter memset and does no work. TRY_BA=1 runs only
# BatchAttention (the persistent kernel) on the first mix: on SM89 its cooperative launch of
# 2 * SMs CTAs with 64-68 KB of shared memory each fails ("too many blocks in cooperative
# launch").
import os
import torch
import flashinfer

HEAD_DIM = 128
PAGE = int(os.environ.get("PAGE", "1"))
TRY_BA = os.environ.get("TRY_BA") is not None
MIXES = [(128, 8192, 1, 4096, 8192), (128, 2048, 1, 1024, 2048), (32, 8192, 1, 4096, 8192),
         (128, 8192, 0, 0, 0), (64, 4096, 2, 2048, 4096)]
MODELS = [("Qwen-8B", 32, 8), ("Llama-3.1-70B", 64, 8)]
dev = "cuda"


def graph_bench(fn, n):
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


def event_bench(fn, n):
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / n


def paged(qo_lens, kv_lens, qo_heads, kv_heads):
    q_indptr = torch.tensor([0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32, device=dev)
    pages = [(k + PAGE - 1) // PAGE for k in kv_lens]
    kv_indptr = torch.tensor([0] + list(torch.tensor(pages).cumsum(0)), dtype=torch.int32, device=dev)
    n = int(kv_indptr[-1])
    q = torch.randn(int(q_indptr[-1]), qo_heads, HEAD_DIM, dtype=torch.float16, device=dev)
    kv = torch.randn(n, 2, PAGE, kv_heads, HEAD_DIM, dtype=torch.float16, device=dev)
    indices = torch.arange(n, dtype=torch.int32, device=dev)
    last = torch.tensor([(k - 1) % PAGE + 1 for k in kv_lens], dtype=torch.int32, device=dev)
    return q_indptr, kv_indptr, indices, last, q, kv


def prefill_time(ws, qo_lens, kv_lens, qo_heads, kv_heads, n):
    qi, ki, idx, last, q, kv = paged(qo_lens, kv_lens, qo_heads, kv_heads)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
    w.plan(qi, ki, idx, last, qo_heads, kv_heads, HEAD_DIM, PAGE, causal=True,
           q_data_type=torch.float16, kv_data_type=torch.float16)
    return graph_bench(lambda: w.run(q, kv), n)


def main():
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    print(f"page size {PAGE}; times in us")
    print("model           decodes  dkv    prefill  pkv   | batch-prefill  decode+prefill  pod(events)")
    for name, qo_heads, kv_heads in MODELS:
        for nd, dkv, np_, pq, pkv in MIXES:
            n = 20
            qo_lens = [1] * nd + [pq] * np_
            kv_lens = [dkv] * nd + [pkv] * np_
            qi, ki, idx, last, q, kv = paged(qo_lens, kv_lens, qo_heads, kv_heads)
            if TRY_BA:
                # BatchAttention: on SM89 the launch fails (and leaves the error pending for
                # the next CUDA call), so it is tried alone.
                ba = flashinfer.BatchAttention(kv_layout="NHD")
                ba.plan(qi, ki, idx, torch.tensor(kv_lens, dtype=torch.int32, device=dev), qo_heads, kv_heads,
                        HEAD_DIM, HEAD_DIM, PAGE, causal=True, q_data_type=torch.float16,
                        kv_data_type=torch.float16)
                ba.run(q, kv)
                torch.cuda.synchronize()
                print(f"{name} {nd}x{dkv} + {np_}x{pq}/{pkv}: BatchAttention {graph_bench(lambda: ba.run(q, kv), n):.1f} us")
                return
            # BatchPrefill over the whole batch.
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
            w.plan(qi, ki, idx, last, qo_heads, kv_heads, HEAD_DIM, PAGE, causal=True,
                   q_data_type=torch.float16, kv_data_type=torch.float16)
            t_whole = graph_bench(lambda: w.run(q, kv), n)
            del q, kv, w
            torch.cuda.empty_cache()
            # Decode + Prefill (separate wrappers, as the benchmark).
            t_sep = 0.0
            if nd:
                qi, ki, idx, last, q, kv = paged([1] * nd, [dkv] * nd, qo_heads, kv_heads)
                d = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend="fa2", use_tensor_cores=True)
                d.plan(ki, idx, last, qo_heads, kv_heads, HEAD_DIM, PAGE,
                       q_data_type=torch.float16, kv_data_type=torch.float16)
                t_sep += graph_bench(lambda: d.run(q, kv), n)
                del q, kv, d
            if np_:
                t_sep += prefill_time(ws, [pq] * np_, [pkv] * np_, qo_heads, kv_heads, n)
            torch.cuda.empty_cache()
            # POD: one (contiguous) prefill + the paged decodes.
            t_pod = "-"
            if np_ == 1 and nd:
                q_p = torch.randn(pq, qo_heads, HEAD_DIM, dtype=torch.float16, device=dev)
                k_p = torch.randn(pkv, kv_heads, HEAD_DIM, dtype=torch.float16, device=dev)
                v_p = torch.randn_like(k_p)
                qi, ki, idx, last, q_d, kv_d = paged([1] * nd, [dkv] * nd, qo_heads, kv_heads)
                pod = flashinfer.PODWithPagedKVCacheWrapper(ws, kv_layout="NHD")
                pod.plan(ki, idx, last_page_len=last, num_qo_heads=qo_heads, num_kv_heads=kv_heads,
                         head_dim=HEAD_DIM, page_size=PAGE, q_data_type=torch.float16,
                         kv_data_type=torch.float16)
                t_pod = f"{event_bench(lambda: pod.run(q_p, k_p, v_p, q_d, kv_d, causal_p=True), n):.1f}"
                del q_p, k_p, v_p, q_d, kv_d, pod
                torch.cuda.empty_cache()
            print(f"{name:15} {nd:7}  {dkv:5}  {np_:3}x{pq:<5} {pkv:5} | {t_whole:13.1f}  "
                  f"{t_sep:14.1f}  {t_pod:>11}")


if __name__ == "__main__":
    main()
