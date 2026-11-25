# FlashInfer gated_delta_rule_decode_pretranspose (single-token Gated Delta Net decode) on the shapes of
# cute-rs's `gdn` example: Qwen3-Next (16 q/k heads, 32 v heads, K = V = 128), bf16 q/k/v/a/b,
# f32 and bf16 state, batch {1, 4, 16, 64, 128}, and the prefill below. Timed from CUDA-graph replays
# (100 launches per graph). `ONLY=decode` or `ONLY=prefill` runs one part.
# FlashInfer's GDN decode is written for SM90+; on older GPUs this script reports the failure.
import os

import torch
import flashinfer
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

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


torch.manual_seed(0)
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
H, HV, K = 16, 32, 128
ONLY = os.environ.get("ONLY", "")


def decode_bench():
    print("state  batch  time(us)    GB/s")
    for sdt, name in ((torch.float32, "f32"), (torch.bfloat16, "bf16")):
        for batch in (1, 4, 16, 64, 128):
            q = torch.randn(batch, 1, H, K, device=dev, dtype=torch.bfloat16)
            k = torch.randn(batch, 1, H, K, device=dev, dtype=torch.bfloat16)
            v = torch.randn(batch, 1, HV, K, device=dev, dtype=torch.bfloat16)
            state = torch.randn(batch, HV, K, K, device=dev).to(sdt)
            A_log = torch.randn(HV, device=dev) * 0.1
            a = torch.randn(batch, 1, HV, device=dev, dtype=torch.bfloat16) * 0.1
            dt_bias = torch.randn(HV, device=dev) * 0.1
            b = torch.randn(batch, 1, HV, device=dev, dtype=torch.bfloat16)
            out = torch.empty(batch, 1, HV, K, device=dev, dtype=torch.bfloat16)

            def fn():
                gated_delta_rule_decode_pretranspose(
                    q=q, k=k, v=v, state=state, A_log=A_log, a=a, dt_bias=dt_bias, b=b,
                    scale=K ** -0.5, output=out, use_qk_l2norm=True)

            try:
                t = graph_bench(fn)
            except Exception as e:
                print(f"{name:5}  {batch:5}  failed: {type(e).__name__}: {str(e)[:200]}")
                continue
            nbytes = 2 * state.numel() * state.element_size() + (q.numel() + k.numel() + 2 * v.numel() + 2 * a.numel()) * 2
            print(f"{name:5}  {batch:5}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")

    # ---- Multi-token (gated_delta_rule_mtp), f32 state pool, bf16 inputs --------------------------------
    from flashinfer.gdn_decode import gated_delta_rule_mtp

    print("mode           T  batch  time(us)    GB/s")
    for mode, update, cached in (("verify", False, False), ("verify+cache", False, True), ("update", True, False)):
        for T in (2, 4):
            for batch in (1, 16, 64):
                q = torch.randn(batch, T, H, K, device=dev, dtype=torch.bfloat16)
                k = torch.randn(batch, T, H, K, device=dev, dtype=torch.bfloat16)
                v = torch.randn(batch, T, HV, K, device=dev, dtype=torch.bfloat16)
                pool = torch.randn(batch, HV, K, K, device=dev)
                idx = torch.arange(batch, dtype=torch.int32, device=dev)
                A_log = torch.randn(HV, device=dev)
                a = torch.randn(batch, T, HV, device=dev, dtype=torch.bfloat16)
                dt_bias = torch.randn(HV, device=dev, dtype=torch.bfloat16)
                b = torch.randn(batch, T, HV, device=dev, dtype=torch.bfloat16)
                out = torch.empty(batch, T, HV, K, device=dev, dtype=torch.bfloat16)
                cache = torch.zeros(batch, T, HV, K, K, device=dev) if cached else None

                def fn():
                    gated_delta_rule_mtp(q, k, v, pool, idx, A_log, a, dt_bias, b, K ** -0.5, out, cache,
                                         disable_state_update=not update, use_qk_l2norm=True)

                try:
                    t = graph_bench(fn)
                except Exception as e:
                    print(f"{mode:12} {T:3}  {batch:5}  failed: {type(e).__name__}: {str(e)[:200]}")
                    continue
                sb = pool[0].numel() * 4 * batch
                nbytes = sb * (1 + int(update) + (T if cached else 0)) + batch * T * (2 * H * K + 2 * HV * K + 2 * HV) * 2
                print(f"{mode:12} {T:3}  {batch:5}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")

# ---- Prefill (chunk_gated_delta_rule): Qwen3-Next heads, bf16 q/k/v, f32 alpha/beta/state ------
# Shapes of cute-rs's `gdn` example (`ONLY=prefill`): 1k, 4k and 16k tokens in one and in four
# sequences, L2 norm in the kernel, initial state in and final state out. On SM8x FlashInfer runs
# its CuTe DSL SM80 kernel (`gdn_kernels/delta_rule_dsl/delta_rule_sm80.py`).
from flashinfer.gdn_prefill import chunk_gated_delta_rule

def extras_bench():
    # Non-transposed [K, V] state (gated_delta_rule_decode, f32 state, T = 1) and the per-token pool
    # scatter (gated_delta_rule_mtp with ssm_state_indices, T = 4, f32 pool).
    from flashinfer.gdn_decode import gated_delta_rule_decode, gated_delta_rule_mtp

    print("mode            T  batch  time(us)    GB/s")
    for batch in (1, 4, 16, 64):
        q = torch.randn(batch, 1, H, K, device=dev, dtype=torch.bfloat16)
        k = torch.randn(batch, 1, H, K, device=dev, dtype=torch.bfloat16)
        v = torch.randn(batch, 1, HV, K, device=dev, dtype=torch.bfloat16)
        state = torch.randn(batch, HV, K, K, device=dev)
        A_log = torch.randn(HV, device=dev) * 0.1
        a = torch.randn(batch, 1, HV, device=dev, dtype=torch.bfloat16) * 0.1
        dt_bias = torch.randn(HV, device=dev) * 0.1
        b = torch.randn(batch, 1, HV, device=dev, dtype=torch.bfloat16)
        out = torch.empty(batch, 1, HV, K, device=dev, dtype=torch.bfloat16)

        def fn():
            gated_delta_rule_decode(q=q, k=k, v=v, state=state, A_log=A_log, a=a, dt_bias=dt_bias, b=b,
                                    scale=K ** -0.5, output=out, use_qk_l2norm=True, backend="flashinfer")

        try:
            t = graph_bench(fn)
        except Exception as e:
            print(f"kv [K,V]       1  {batch:5}  failed: {type(e).__name__}: {str(e)[:200]}")
            continue
        nbytes = 2 * state.numel() * 4 + (q.numel() + k.numel() + 2 * v.numel() + 2 * a.numel()) * 2
        print(f"kv [K,V]       1  {batch:5}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")
    T = 4
    for batch in (1, 4, 16, 64):
        q = torch.randn(batch, T, H, K, device=dev, dtype=torch.bfloat16)
        k = torch.randn(batch, T, H, K, device=dev, dtype=torch.bfloat16)
        v = torch.randn(batch, T, HV, K, device=dev, dtype=torch.bfloat16)
        pool = torch.randn(batch * (T + 1), HV, K, K, device=dev)
        idx = torch.arange(batch, dtype=torch.int32, device=dev)
        ssm = torch.arange(batch, batch * (T + 1), dtype=torch.int32, device=dev).reshape(batch, T)
        A_log = torch.randn(HV, device=dev)
        a = torch.randn(batch, T, HV, device=dev, dtype=torch.bfloat16)
        dt_bias = torch.randn(HV, device=dev, dtype=torch.bfloat16)
        b = torch.randn(batch, T, HV, device=dev, dtype=torch.bfloat16)
        out = torch.empty(batch, T, HV, K, device=dev, dtype=torch.bfloat16)

        def fn():
            gated_delta_rule_mtp(q, k, v, pool, idx, A_log, a, dt_bias, b, K ** -0.5, out, None,
                                 disable_state_update=False, use_qk_l2norm=True, ssm_state_indices=ssm)

        try:
            t = graph_bench(fn)
        except Exception as e:
            print(f"scatter        {T}  {batch:5}  failed: {type(e).__name__}: {str(e)[:200]}")
            continue
        sb = HV * K * K * 4 * batch
        nbytes = sb * (1 + T) + batch * T * (2 * H * K + 2 * HV * K + 2 * HV) * 2
        print(f"scatter        {T}  {batch:5}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")


if ONLY == "extras":
    extras_bench()
    raise SystemExit
if ONLY != "prefill":
    decode_bench()
if ONLY != "decode":
    print("prefill tokens seqs  use_cp  time(us)    GB/s")
    for total in (1024, 4096, 16384):
        for seqs in (1, 4):
            q = torch.rand(total, H, K, device=dev, dtype=torch.bfloat16) * 2 - 1
            k = torch.rand(total, H, K, device=dev, dtype=torch.bfloat16) * 2 - 1
            v = torch.rand(total, HV, K, device=dev, dtype=torch.bfloat16) * 2 - 1
            g = torch.rand(total, HV, device=dev) * 0.15 + 0.85
            beta = torch.rand(total, HV, device=dev)
            s0 = (torch.rand(seqs, HV, K, K, device=dev) - 0.5)
            s1 = torch.empty_like(s0)
            out = torch.empty(total, HV, K, device=dev, dtype=torch.bfloat16)
            cu = torch.arange(0, total + 1, total // seqs, dtype=torch.int32, device=dev)
            for use_cp in (False, "auto"):

                def fn():
                    chunk_gated_delta_rule(q, k, v, g, beta, K ** -0.5, s0, True, cu, True, out, s1, use_cp=use_cp)

                try:
                    t = graph_bench(fn)
                except Exception as e:
                    print(f"prefill {total:6} {seqs:4}  {str(use_cp):6}  failed: {type(e).__name__}: {str(e)[:300]}")
                    continue
                nbytes = total * (2 * H * K + 2 * HV * K) * 2 + total * HV * 8 + 2 * s0.numel() * 4
                print(f"prefill {total:6} {seqs:4}  {str(use_cp):6}  {t:8.1f}  {nbytes / (t * 1e-6) / 1e9:6.1f}")
