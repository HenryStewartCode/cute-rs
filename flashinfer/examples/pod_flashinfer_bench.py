# FlashInfer's POD attention (PODWithPagedKVCacheWrapper, fa2) on the single-prefill shapes
# of its benchmarks/bench_mixed_attention.py, in f16 (cute-rs's `pod` example's shapes):
# 128 decodes next to one causal prefill, 32 query / 8 KV heads, head dim 128, page size 1,
# `arange` page tables. Times (microseconds) are CUDA-graph replays, except the POD launch
# (CUDA events; *its graph replay does no work, see event_bench): the POD launch, the
# sequential single prefill + tensor-core batch decode (FlashInfer's comparison), and
# single prefill + its CUDA-core batch decode.
import torch
import flashinfer

HEAD_DIM, KV_HEADS, QO_HEADS, DECODES = 128, 8, 32, 128
SHAPES = [(512, 512, 2048), (1536, 1536, 2048), (2048, 2048, 2048), (4096, 4096, 4096),
          (4096, 4096, 8192), (6000, 7000, 8192)]
dev = "cuda"


def graph_bench(fn, n):
    # GPU time alone: the launches replayed from a CUDA graph (falls back to the
    # profiler's summed kernel time if the run cannot be captured).
    try:
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
        return start.elapsed_time(end) * 1000 / n, "graph"
    except Exception as e:
        torch.cuda.synchronize()
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
        us = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
        return us / n, "prof"


def event_bench(fn, n):
    # Launches back to back between two CUDA events (FlashInfer's POD clears its scheduler
    # counters with a synchronous cudaMemset, which a CUDA graph does not replay, so a
    # replayed POD kernel finds its counters exhausted and does no work).
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / n


def main():
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    print("prefill qo/kv  decode kv    pod(us)  pod-graph(us)*  seq-tc(us)  prefill+decode(us)")
    for qo, kv, dkv in SHAPES:
        n = 20 if dkv >= 8192 else 50
        q_p = torch.randn(qo, QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
        k_p = torch.randn(kv, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
        v_p = torch.randn_like(k_p)
        q_d = torch.randn(DECODES, QO_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
        pages = DECODES * dkv
        kv_d = torch.randn(pages, 2, 1, KV_HEADS, HEAD_DIM, dtype=torch.float16, device=dev)
        indptr = torch.arange(0, DECODES + 1, device=dev, dtype=torch.int32) * dkv
        indices = torch.arange(0, pages, device=dev, dtype=torch.int32)
        last = torch.ones(DECODES, device=dev, dtype=torch.int32)

        pod = flashinfer.PODWithPagedKVCacheWrapper(ws, kv_layout="NHD")
        pod.plan(indptr, indices, last_page_len=last, num_qo_heads=QO_HEADS, num_kv_heads=KV_HEADS,
                 head_dim=HEAD_DIM, page_size=1, q_data_type=torch.float16, kv_data_type=torch.float16)
        run_pod = lambda: pod.run(q_p, k_p, v_p, q_d, kv_d, causal_p=True)
        o_p, o_d = run_pod()

        prefill = lambda: flashinfer.prefill.single_prefill_with_kv_cache(
            q_p, k_p, v_p, causal=True, pos_encoding_mode="NONE", backend="fa2")
        decs = []
        for tc in (True, False):
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
            w.plan(indptr, indices, last, QO_HEADS, KV_HEADS, HEAD_DIM, 1,
                   q_data_type=torch.float16, kv_data_type=torch.float16)
            decs.append(w)
        torch.testing.assert_close(o_p, prefill(), rtol=4e-3, atol=4e-3)
        torch.testing.assert_close(o_d, decs[0].run(q_d, kv_d), rtol=4e-3, atol=4e-3)

        t_pod_graph, _ = graph_bench(run_pod, n)
        t_pod = event_bench(run_pod, n)
        t_tc, _ = graph_bench(lambda: (prefill(), decs[0].run(q_d, kv_d)), n)
        t_cc, _ = graph_bench(lambda: (prefill(), decs[1].run(q_d, kv_d)), n)
        print(f"{qo:5}/{kv:<5}  {dkv:9}  {t_pod:9.1f}  {t_pod_graph:15.1f}  {t_tc:10.1f}  {t_cc:18.1f}")
        del kv_d, pod, decs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
