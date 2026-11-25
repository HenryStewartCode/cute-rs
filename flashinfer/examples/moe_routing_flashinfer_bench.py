# FlashInfer 0.7.5's MoE routing on SM89, on the shapes of cute-rs's flashinfer moe example:
# fused_topk_deepseek (DeepSeek-V3: 256 experts, 8 groups, top-4 groups, top-8, scaling 2.5)
# with f32 and bf16 logits. The softmax-top-k and
# Renormalize methods exist in FlashInfer only as trtllm-gen routing (SM100+); the script
# shows that it refuses SM89 and times PyTorch's softmax + topk instead, for scale only.
# GPU time from CUDA-graph replays.
import torch
import torch.nn.functional as F
from flashinfer.fused_moe import fused_topk_deepseek
from flashinfer.tllm_enums import RoutingMethodType

dev = "cuda"
TOKENS = [1, 16, 128, 1024, 8192]


def graph_bench(fn, n=100):
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


def ds_reference(logits, bias, n_group, topk_group, k, scale):
    s = logits.float().sigmoid()
    b = s + bias.float()
    t, e = b.shape
    g = b.view(t, n_group, e // n_group).topk(2, dim=-1).values.sum(-1)
    keep = g.topk(topk_group, dim=-1).indices
    mask = torch.zeros(t, n_group, device=dev, dtype=torch.bool).scatter_(1, keep, True)
    mask = mask.repeat_interleave(e // n_group, dim=1)
    ids = b.masked_fill(~mask, float("-inf")).topk(k, dim=-1).indices
    w = s.gather(1, ids)
    return ids, w / (w.sum(-1, keepdim=True) + 1e-20) * scale


try:
    from flashinfer.fused_moe.trtllm_gen_routing import trtllm_gen_routing

    trtllm_gen_routing(torch.randn(4, 128, device=dev), None, RoutingMethodType.Renormalize, 8)
    print("trtllm_gen_routing runs on this GPU")
except Exception as e:  # noqa: BLE001
    print("trtllm_gen_routing on this GPU: %s" % str(e).splitlines()[0][:160])

gen = torch.Generator(device=dev)
gen.manual_seed(0)
print("route                 scores experts top_k tokens  time(us)")
# Kimi K2 (n_group = topk_group = 1) is rejected by 0.7.5's argument check
# (`topk_group * n_group >= topk`), so only DeepSeek-V3 is timed.
for name, e, n_group, topk_group, k in [("deepseek_v3", 256, 8, 4, 8)]:
    for dt in [torch.float32, torch.bfloat16]:
        bias = (torch.rand(e, device=dev, generator=gen) * 0.2 - 0.1).to(dt)
        for t in TOKENS:
            logits = (torch.randn(t, e, device=dev, generator=gen) * 1.2).to(dt)
            vals = torch.empty(t, k, dtype=dt, device=dev)
            idx = torch.empty(t, k, dtype=torch.int32, device=dev)
            run = lambda: fused_topk_deepseek(logits, bias, n_group, topk_group, k, 2.5, vals, idx)
            run()
            if t == 128:
                rid, rw = ds_reference(logits, bias, n_group, topk_group, k, 2.5)
                same = (idx.sort(-1).values.long() == rid.sort(-1).values).float().mean().item()
                print("  check: %.4f of selections match torch" % same)
            us = graph_bench(run)
            print("%-21s %6s %7d %5d %6d %9.2f" % (name, str(dt).split(".")[1].replace("float32", "f32").replace("bfloat16", "bf16"), e, k, t, us))

print("torch (not FlashInfer) softmax + topk [+ renormalize], f32 logits")
for name, e, k in [("mixtral_softmax", 8, 2), ("qwen3_softmax_renorm", 128, 8)]:
    for t in TOKENS:
        logits = torch.randn(t, e, device=dev, generator=gen)

        def run():
            w, i = torch.topk(F.softmax(logits, dim=-1), k, dim=-1)
            return w / w.sum(-1, keepdim=True), i

        print("%-21s %6s %7d %5d %6d %9.2f" % (name, "f32", e, k, t, graph_bench(run)))


# The trtllm-gen methods FlashInfer has no SM89 kernel for: PyTorch references, for scale only.
def ref_method(method, logits, bias, k):
    lf = logits.float()
    if method == "topk":
        return torch.topk(lf, k, dim=-1)
    if method == "topk_sigmoid":
        v, i = torch.topk(lf, k, dim=-1)
        return v.sigmoid(), i
    s = lf.sigmoid()
    if method == "minimax2":
        i = torch.topk(s + bias.float(), k, dim=-1).indices
        w = s.gather(1, i)
        return w / (w.sum(-1, keepdim=True) + 1e-20), i
    w, i = torch.topk(s, k, dim=-1)
    if method == "sigmoid_renorm":
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w, i


print("torch (not FlashInfer) references of the other trtllm-gen methods, f32 logits")
for name, method, e, k in [
    ("llama4_128e", "sigmoid", 128, 1),
    ("topk_128e", "topk", 128, 8),
    ("sigmoid_128e", "sigmoid", 128, 8),
    ("sigmoid_renorm_128e", "sigmoid_renorm", 128, 8),
    ("minimax2_256e", "minimax2", 256, 8),
    ("topk_sigmoid_128e", "topk_sigmoid", 128, 8),
]:
    bias = torch.rand(e, device=dev, generator=gen) * 0.2 - 0.1
    for t in TOKENS:
        logits = torch.randn(t, e, device=dev, generator=gen)
        print("%-21s %6s %7d %5d %6d %9.2f" % (name, "f32", e, k, t, graph_bench(lambda: ref_method(method, logits, bias, k))))
