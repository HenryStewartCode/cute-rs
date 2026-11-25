# FlashInfer merge_state / merge_state_in_place / merge_states on the shapes of cute-rs's
# `cascade` example (seq {1, 64, 4096} x heads {32, 64}, head_dim 128, f16; merge_states of
# 4 and 64 states), timed from CUDA-graph replays of one L2-resident set of buffers: 100
# launches per graph, one replay between two CUDA events.
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


torch.manual_seed(0)
print("flashinfer", flashinfer.__version__, torch.cuda.get_device_name())
print("op                    sets  seq   heads  time(us)    GB/s")
for op in ("merge_state", "merge_state_in_place", "merge_states"):
    for heads in (32, 64):
        for sets in ((4, 64) if op == "merge_states" else (2,)):
            for seq in ((1, 64) if sets == 64 else (1, 64, 4096)):
                hd = 128
                v = lambda *s: torch.randn(*s, dtype=torch.float16, device=dev)
                s = lambda *sh: torch.randn(*sh, dtype=torch.float32, device=dev) * 3
                vb, sb = v(seq, heads, hd), s(seq, heads)
                vo, so = v(seq, heads, hd), s(seq, heads)
                vm, sm = v(seq, sets, heads, hd), s(seq, sets, heads)
                vs, ss = v(seq, heads, hd), s(seq, heads)
                if op == "merge_state":
                    fn = lambda: flashinfer.merge_state(vb, sb, vo, so)
                    nbytes = seq * heads * (3 * hd * 2 + 3 * 4)
                elif op == "merge_state_in_place":
                    fn = lambda: flashinfer.merge_state_in_place(vs, ss, vo, so)
                    nbytes = seq * heads * (3 * hd * 2 + 3 * 4)
                else:
                    fn = lambda: flashinfer.merge_states(vm, sm)
                    nbytes = seq * heads * (sets + 1) * (hd * 2 + 4)
                t = graph_bench(fn)
                print(f"{op:21} {sets:4}  {seq:5} {heads:5}  {t:8.2f}  {nbytes / (t * 1e-6) / 1e9:7.1f}")
