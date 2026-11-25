# SPDX-License-Identifier: Apache-2.0
# FlashInfer's block-sparse wrappers (fa2) on the shapes of cute-rs's `sparse` example, GPU
# time from CUDA-graph replays. The block patterns use the example's `keep_block` hash, so
# both sides run the same blocks.
import sys

import torch
import flashinfer

dev = "cuda"
M64 = (1 << 64) - 1


def keep_block(h, i, j, density):
    x = (((h * 1_000_003 + i) * 1_000_003 + j) * 0x9E3779B97F4A7C15) & M64
    x ^= x >> 29
    x = (x * 0xBF58476D1CE4E5B9) & M64
    x ^= x >> 32
    return x % 1000 < round(density * 1000)


def graph_bench(fn, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def iters_for(flops):
    return max(5, min(100, int(4e11 / flops)))


def bsr(seq, hd, density, mask, hq=32, hkv=8, R=64, C=64):
    mb, nb = seq // R, seq // C
    indptr, indices = [0], []
    for i in range(mb):
        indices += [j for j in range(nb) if keep_block(0, i, j, density)]
        indptr.append(len(indices))
    indptr = torch.tensor(indptr, dtype=torch.int32, device=dev)
    indices = torch.tensor(indices, dtype=torch.int32, device=dev)
    nnz = indices.numel()
    q = torch.randn(seq, hq, hd, dtype=torch.half, device=dev)
    k = torch.randn(seq, hkv, hd, dtype=torch.half, device=dev)
    v = torch.randn(seq, hkv, hd, dtype=torch.half, device=dev)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.sparse.BlockSparseAttentionWrapper(ws, backend="fa2")
    m = torch.rand(nnz, R, C, device=dev) < 0.5 if mask else None
    w.plan(indptr, indices, seq, seq, R, C, hq, hkv, hd, mask=m, q_data_type=torch.half)
    flops = 4.0 * nnz * R * C * hq * hd
    t = graph_bench(lambda: w.run(q, k, v), iters_for(flops))
    print(f"{seq:6} {hd:4}  {density:7}  {str(mask):5} {nnz:5}  {t:10.1f}  {flops / (t * 1e-6) / 1e12:7.2f}", flush=True)


def variable(seq, density, hq=32, hkv=32, hd=128, mb=20, nb=50):
    def sizes(blocks):
        s = [seq // blocks] * blocks
        s[-1] = seq - sum(s[:-1])
        return torch.tensor(s, dtype=torch.int32).unsqueeze(0).repeat(hkv, 1)

    rows, cols = sizes(mb), sizes(nb)
    mask = torch.tensor(
        [[[keep_block(h, i, j, density) for j in range(nb)] for i in range(mb)] for h in range(hkv)], dtype=torch.bool
    )
    q = torch.randn(hq, seq, hd, dtype=torch.half, device=dev)
    k = torch.randn(hkv, seq, hd, dtype=torch.half, device=dev)
    v = torch.randn(hkv, seq, hd, dtype=torch.half, device=dev)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.sparse.VariableBlockSparseAttentionWrapper(ws, backend="fa2")
    w.plan(mask.to(dev), rows.to(dev), cols.to(dev), hq, hkv, hd, q_data_type=torch.half)
    kv_tokens = int((mask * cols[:, None, :]).sum())
    pairs = sum(int(rows[h, i]) * int((mask[h, i] * cols[h]).sum()) for h in range(hkv) for i in range(mb))
    flops = 4.0 * pairs * (hq // hkv) * hd
    t = graph_bench(lambda: w.run(q, k, v), iters_for(flops))
    print(f"{seq:6}  {density:7}  {kv_tokens:12}  {t:10.1f}  {flops / (t * 1e-6) / 1e12:7.2f}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(torch.cuda.get_device_name(), "flashinfer", flashinfer.__version__)
    if what in ("all", "bsr"):
        print("BlockSparseAttentionWrapper (fa2), 64 x 64 blocks, 32/8 heads")
        print("  seq   hd  density  mask   nnz     gpu(us)   TFLOPS")
        for seq in [4096, 8192, 16384]:
            for density in [0.5, 0.1]:
                bsr(seq, 128, density, False)
        bsr(8192, 64, 0.5, False)
        bsr(8192, 64, 0.1, False)
        bsr(4096, 128, 0.5, True)
    if what in ("all", "variable"):
        print("VariableBlockSparseAttentionWrapper (fa2), 20 x 50 blocks, 32/32 heads, head dim 128")
        print("  seq  density     kv tokens     gpu(us)   TFLOPS")
        for seq in [8192, 16384]:
            for density in [0.5, 0.1]:
                variable(seq, density)
