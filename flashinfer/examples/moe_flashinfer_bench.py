# FlashInfer 0.7.5's cutlass_fused_moe (bf16/f16 activations and weights, SwiGLU, top-k routing
# given as token_selected_experts + token_final_scales) on SM89: the SM80 fused-SwiGLU grouped
# GEMM path (Fused_Moe_Kernel_sm80 for FC1, CUTLASS MoeFCGemm for FC2). Shapes of cute-rs's
# flashinfer moe example: Mixtral 8x7B (hidden 4096, inter 14336, 8 experts, top-2) and the
# two configs of FlashInfer's benchmarks/bench_cutlass_fused_moe.py (hidden 7168, top-8:
# 256 experts of inter 256, 32 experts of inter 2048). GPU time from CUDA-graph replays.
import os
import torch
import torch.nn.functional as F
import flashinfer.fused_moe as fused_moe
from flashinfer.autotuner import autotune

dev = "cuda"
dtype = torch.bfloat16 if os.environ.get("DTYPE", "bf16") == "bf16" else torch.float16


def graph_bench(fn, n=20):
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


def routing(tokens, e, k, gen):
    logits = torch.randn(tokens, e, device=dev, generator=gen)
    w = F.softmax(logits.float(), dim=-1)
    w, sel = torch.topk(w, k, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    return sel.to(torch.int32), w.float()


def reference(x, sel, scales, w1, w2):
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=dev)
    inter = w2.shape[2]
    for e in range(w1.shape[0]):
        tok, slot = (sel == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        a = x[tok].float()
        up = a @ w1[e, :inter].float().t()
        gate = a @ w1[e, inter:].float().t()
        h = (F.silu(gate) * up).to(dtype).float()
        y = (h @ w2[e].float().t()).to(dtype).float()
        out.index_add_(0, tok, y * scales[tok, slot, None])
    return out


def run(x, sel, scales, w1, w2, out):
    return fused_moe.cutlass_fused_moe(x, sel, scales, w1, w2, dtype, quant_scales=None, output=out)


gen = torch.Generator(device=dev)
gen.manual_seed(0)

# Correctness of FlashInfer itself on a small shape.
x = torch.randn(37, 256, device=dev, generator=gen).to(dtype)
w1 = (torch.randn(4, 2 * 384, 256, device=dev, generator=gen) / 16).to(dtype)
w2 = (torch.randn(4, 256, 384, device=dev, generator=gen) / 16).to(dtype)
sel, sc = routing(37, 4, 2, gen)
out = torch.empty(37, 256, dtype=dtype, device=dev)
run(x, sel, sc, w1, w2, out)
ref = reference(x, sel, sc, w1, w2)
print("check: max abs err %.4f (max |ref| %.3f)" % ((out.float() - ref).abs().max().item(), ref.abs().max().item()))

configs = [
    ("mixtral", 4096, 14336, 8, 2),
    ("e256_i256", 7168, 256, 256, 8),
    ("e32_i2048", 7168, 2048, 32, 8),
]
only = os.environ.get("CONFIGS")
tokens_list = [int(t) for t in os.environ.get("TOKENS", "1,16,128,1024").split(",")]
print("config        hidden  inter experts top_k tokens  time(us)  TFLOPS")
for name, h, inter, e, k in configs:
    if only and name not in only.split(","):
        continue
    w1 = (torch.randn(e, 2 * inter, h, device=dev, generator=gen) / 64).to(dtype)
    w2 = (torch.randn(e, h, inter, device=dev, generator=gen) / 64).to(dtype)
    for tokens in tokens_list:
        x = torch.randn(tokens, h, device=dev, generator=gen).to(dtype)
        sel, sc = routing(tokens, e, k, gen)
        out = torch.empty(tokens, h, dtype=dtype, device=dev)
        with torch.inference_mode(), autotune(True):
            run(x, sel, sc, w1, w2, out)
        t = graph_bench(lambda: run(x, sel, sc, w1, w2, out))
        flops = 2 * tokens * k * h * inter * 3
        print("%-12s %7d %6d %7d %5d %6d %9.1f %7.2f" % (name, h, inter, e, k, tokens, t, flops / t * 1e-6))
    del w1, w2
    torch.cuda.empty_cache()
