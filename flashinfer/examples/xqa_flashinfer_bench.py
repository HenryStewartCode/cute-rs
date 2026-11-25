# FlashInfer's XQA decode (flashinfer.xqa, TensorRT-LLM's mha.cu GQA kernel) on the shapes of
# cute-rs's `xqa` example, timed from CUDA-graph replays (GPU time), as `graph_bench` in
# fi_bench_decode.py.
#
# FlashInfer 0.7.5 refuses XQA below SM90 (`xqa()` raises, and its JIT spec compiles only for
# majors 9/10/12), although mha.cu itself has an SM80/86/89 path. This script therefore builds
# the same sources (mha.cu, xqa_wrapper.cu, flashinfer_xqa_binding.cu) with FlashInfer's own
# flags for sm_89 and USE_SM90_MHA=0, and calls the module's `xqa_wrapper` directly.
import itertools
import sys

import torch
import flashinfer
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import gen_jit_spec
from flashinfer.jit.xqa import xqa_nvcc_flags

HEAD_DIM, KV_HEADS, PAGE = 128, 8, 16
dev = "cuda"
_modules = {}


def module(q_dtype, kv_dtype, group, page, window):
    key = (q_dtype, kv_dtype, group, page, window)
    if key in _modules:
        return _modules[key]
    flags = list(xqa_nvcc_flags) + [
        "-gencode=arch=compute_89,code=sm_89",
        f"-DTOKENS_PER_PAGE={page}",
        f"-DHEAD_ELEMS={HEAD_DIM}",
        "-DINPUT_FP16=1" if q_dtype == torch.float16 else "-DINPUT_FP16=0",
        "-DDTYPE=__half" if q_dtype == torch.float16 else "-DDTYPE=__nv_bfloat16",
        "-DCACHE_ELEM_ENUM=2" if kv_dtype == torch.float8_e4m3fn else "-DCACHE_ELEM_ENUM=0",
        f"-DHEAD_GRP_SIZE={group}",
        f"-DSLIDING_WINDOW={int(window)}",
        "-DLOW_PREC_OUTPUT=0",
        "-DSPEC_DEC=0",
        "-DMLA_WRAPPER=0",
        "-DUSE_SM90_MHA=0",
    ]
    qn = "f16" if q_dtype == torch.float16 else "bf16"
    kn = "e4m3" if kv_dtype == torch.float8_e4m3fn else qn
    spec = gen_jit_spec(
        f"xqa_sm89_{qn}_{kn}_p{page}_g{group}_w{int(window)}",
        [
            jit_env.FLASHINFER_CSRC_DIR / "xqa/mha.cu",
            jit_env.FLASHINFER_CSRC_DIR / "xqa/xqa_wrapper.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_xqa_binding.cu",
        ],
        extra_cuda_cflags=flags,
        extra_ldflags=["-lcuda"],
    )
    m = spec.build_and_load()
    _modules[key] = m
    return m


def problem(batch, kv_len, group, q_dtype, kv_dtype, page=PAGE):
    pages_per = (kv_len + page - 1) // page
    num_pages = batch * pages_per + 2
    gen = torch.float16 if kv_dtype == torch.float8_e4m3fn else kv_dtype
    k = torch.randn(num_pages, page, KV_HEADS, HEAD_DIM, dtype=gen, device=dev).to(kv_dtype)
    v = torch.randn(num_pages, page, KV_HEADS, HEAD_DIM, dtype=gen, device=dev).to(kv_dtype)
    perm = torch.randperm(num_pages, device=dev, dtype=torch.int32)
    table = perm[: batch * pages_per].reshape(batch, pages_per).contiguous()
    seq_lens = torch.full((batch, 1), kv_len, dtype=torch.uint32, device=dev)
    q = torch.randn(batch, 1, KV_HEADS * group, HEAD_DIM, dtype=q_dtype, device=dev)
    return q, k, v, table, seq_lens


def runner(m, q, k, v, table, seq_lens, page, kv_scale=1.0, window=0):
    batch = q.shape[0]
    out = torch.empty_like(q)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    nb_seq = batch * KV_HEADS
    sem = torch.zeros(((nb_seq + 1) // 2) * 2 + 2 + nb_seq + 2, dtype=torch.uint32, device=dev)
    scratch = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    max_seq = table.shape[1] * page

    def fn():
        m.xqa_wrapper(False, sms, KV_HEADS, window, 1.0, None, out, 1.0, q, None, k, v, None, None,
                      table, max_seq, seq_lens, batch, kv_scale, None, 1, None, None, sem, scratch, False)

    return fn, out


def reference(q, k, v, table, seq_lens, b, group, page, kv_scale=1.0, window=0):
    n = int(seq_lens[b, 0])
    kk = k[table[b].long()].reshape(-1, KV_HEADS, HEAD_DIM)[:n].float() * kv_scale
    vv = v[table[b].long()].reshape(-1, KV_HEADS, HEAD_DIM)[:n].float() * kv_scale
    if window and n > window:
        kk, vv = kk[n - window:], vv[n - window:]
    kk = kk.repeat_interleave(group, dim=1)
    vv = vv.repeat_interleave(group, dim=1)
    s = torch.einsum("hd,nhd->hn", q[b, 0].float(), kk) / HEAD_DIM ** 0.5
    return torch.einsum("hn,nhd->hd", s.softmax(-1), vv)


def graph_bench(fn, reps=100):
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
f16, bf16, e4m3 = torch.float16, torch.bfloat16, torch.float8_e4m3fn
# Correctness of the FlashInfer side against PyTorch.
for qd, kd, group, window in [] if "--no-check" in sys.argv else [(f16, f16, 4, 0), (f16, e4m3, 8, 0), (bf16, bf16, 8, 0), (bf16, e4m3, 4, 0),
                              (f16, f16, 8, 300)]:
    m = module(qd, kd, group, PAGE, window > 0)
    q, k, v, table, seq_lens = problem(3, 1000, group, qd, kd)
    seq_lens[1, 0] = 77
    scale = 0.5 if kd == e4m3 else 1.0
    fn, out = runner(m, q, k, v, table, seq_lens, PAGE, scale, window)
    fn()
    torch.cuda.synchronize()
    err = max((out[b, 0].float() - reference(q, k, v, table, seq_lens, b, group, PAGE, scale, window)).abs().max().item()
              for b in range(3))
    print(f"check q {qd} kv {kd} group {group} window {window}: max |o - ref| = {err:.2e}")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
groups = [int(x) for x in args[0].split(",")] if args else [4, 8]
print("q     kv    group  batch  kv_len  graph(us)  GB/s (K+V read)")
for (qd, kd), group, batch, kv_len in itertools.product([(f16, f16), (f16, e4m3), (bf16, bf16)], groups,
                                                        (1, 16, 64), (1024, 4096, 16384)):
    m = module(qd, kd, group, PAGE, False)
    q, k, v, table, seq_lens = problem(batch, kv_len, group, qd, kd)
    fn, _ = runner(m, q, k, v, table, seq_lens, PAGE)
    tg = min(graph_bench(fn) for _ in range(3))
    gbs = 2 * batch * kv_len * KV_HEADS * HEAD_DIM * k.element_size() / (tg * 1e-6) / 1e9
    print(f"{str(qd)[6:]:5} {str(kd)[6:]:5} {group:5}  {batch:5}  {kv_len:6}  {tg:9.1f}  {gbs:8.1f}")
    del q, k, v, table, seq_lens, fn
    torch.cuda.empty_cache()
