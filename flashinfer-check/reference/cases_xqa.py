# SPDX-License-Identifier: Apache-2.0
"""`flashinfer.xqa`: TensorRT-LLM's XQA GQA decode over a paged cache.

FlashInfer 0.7.5's `xqa()` refuses GPUs below SM90 (and its JIT spec builds only for SM90, 100 and
120), though the kernel source has an SM80/89 path. So the reference builds FlashInfer's own XQA
sources for sm_89 through FlashInfer's JIT (`gen_jit_spec`, the same sources and flags as its
SM90 build, plus `USE_SM90_MHA=0` and an sm_89 target) and calls the built module's `xqa_wrapper`
with the arguments `flashinfer.xqa()` gives it.

Attention outputs add up many terms: Float16 `close:1e-3:2e-3`, BFloat16 `close:8e-3:1.6e-2`.
"""
import torch

from common import Case

DEV = "cuda"
HD = 128
F16, BF16, E4 = (torch.float16, "f16"), (torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "e4m3")
OUT_RULE = {"f16": "close:1e-3:2e-3", "bf16": "close:8e-3:1.6e-2"}
_MODULES = {}
_BUFFERS = {}


def module(qtype, kvtype, group, page, window):
    key = (qtype, kvtype, group, page, window)
    if key in _MODULES:
        return _MODULES[key]
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec
    from flashinfer.jit.xqa import xqa_nvcc_flags

    f16 = qtype == torch.float16
    fp8 = kvtype == torch.float8_e4m3fn
    flags = list(xqa_nvcc_flags) + [
        "-gencode=arch=compute_89,code=sm_89",
        f"-DTOKENS_PER_PAGE={page}",
        f"-DHEAD_ELEMS={HD}",
        f"-DINPUT_FP16={int(f16)}",
        "-DDTYPE=__half" if f16 else "-DDTYPE=__nv_bfloat16",
        "-DCACHE_ELEM_ENUM=2" if fp8 else "-DCACHE_ELEM_ENUM=0",
        f"-DHEAD_GRP_SIZE={group}",
        f"-DSLIDING_WINDOW={int(window)}",
        "-DLOW_PREC_OUTPUT=0",
        "-DSPEC_DEC=0",
        "-DMLA_WRAPPER=0",
        "-DUSE_SM90_MHA=0",
    ]
    qn = "f16" if f16 else "bf16"
    kn = "e4m3" if fp8 else qn
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
    _MODULES[key] = spec.build_and_load()
    return _MODULES[key]


def buffers():
    if not _BUFFERS:
        _BUFFERS["scratch"] = torch.zeros(256 << 20, dtype=torch.uint8, device=DEV)
    return _BUFFERS["scratch"]


def xqa_case(name, qdt, kvdt, group, page, hkv, lens, q_scale=1.0, kv_scale=1.0, window=0, sinks=False):
    qtype, qtag = qdt
    kvtype, kvtag = kvdt
    batch = len(lens)
    hqo = hkv * group
    pages = [-(-n // page) for n in lens]
    max_pages = max(pages)
    num_pages = sum(pages) + 2
    perm = torch.randperm(num_pages, dtype=torch.int32, device=DEV)
    table = torch.zeros(batch, max_pages, dtype=torch.int32, device=DEV)
    off = 0
    for b, n in enumerate(pages):
        table[b, :n] = perm[off : off + n]
        off += n
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=DEV)
    q = torch.randn(batch, 1, hqo, HD, device=DEV).to(qtype)
    k = torch.randn(num_pages, page, hkv, HD, device=DEV).to(kvtype)
    v = torch.randn(num_pages, page, hkv, HD, device=DEV).to(kvtype)
    assert k.numel() * k.element_size() < 45e6, name
    c = Case(f"xqa_{name}", "xqa", dtype=qtag, kv_dtype=kvtag, group=group, page_size=page, kv_heads=hkv,
             seq_lens=lens, q_scale=q_scale, kv_scale=kv_scale, window=window, sinks=sinks)
    for n, t in (("q", q), ("k_cache", k), ("v_cache", v), ("page_table", table), ("seq_lens", seq_lens)):
        c.input(n, t)
    sink_t = None
    if sinks:
        sink_t = torch.randn(hqo, dtype=torch.float32, device=DEV) * 2
        c.input("sinks", sink_t)

    m = module(qtype, kvtype, group, page, window > 0)
    out = torch.zeros_like(q)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    nb_seq = batch * hkv
    sem = torch.zeros(((nb_seq + 1) // 2) * 2 + 2 + nb_seq + 2, dtype=torch.uint32, device=DEV)
    # The arguments are those `flashinfer.xqa()` passes (q_seq_len 1, no mask, PDL off).
    m.xqa_wrapper(False, sms, hkv, window, q_scale, None, out, 1.0, q, sink_t, k, v, None, None,
                  table, max_pages * page, seq_lens.to(torch.uint32).view(batch, 1),
                  batch, kv_scale, None, 1, None, None, sem, buffers(), False)
    torch.cuda.synchronize()
    c.output("out", out, OUT_RULE[qtag])
    return c


MIXED = [1, 7, 300, 1500, 4096, 33, 2049, 1]


def cases():
    yield xqa_case("f16_g4_p16_mixed", F16, F16, 4, 16, 8, MIXED)
    yield xqa_case("f16_g4_p16_batch1", F16, F16, 4, 16, 8, [3000])
    yield xqa_case("f16_g4_p16_qscale", F16, F16, 4, 16, 8, MIXED, q_scale=0.7)
    yield xqa_case("f16_g4_p16_sinks", F16, F16, 4, 16, 8, MIXED, sinks=True)
    yield xqa_case("bf16_g8_p16_mixed", BF16, BF16, 8, 16, 4, MIXED)
    yield xqa_case("bf16_g8_p16_batch1", BF16, BF16, 8, 16, 4, [1000])
    yield xqa_case("f16_e4m3_g8_p16_mixed", F16, E4, 8, 16, 4, MIXED, kv_scale=0.5)
    yield xqa_case("f16_e4m3_g8_p16_batch1", F16, E4, 8, 16, 4, [2500], kv_scale=1.5, q_scale=0.8)
    yield xqa_case("bf16_e4m3_g4_p32_mixed", BF16, E4, 4, 32, 8, MIXED, kv_scale=2.0)
    yield xqa_case("f16_g1_p64_mixed", F16, F16, 1, 64, 8, MIXED)
    yield xqa_case("f16_g16_p128_mixed", F16, F16, 16, 128, 2, MIXED)
    yield xqa_case("f16_g8_p16_window", F16, F16, 8, 16, 4, MIXED, window=300)
    yield xqa_case("f16_g8_p16_window_short", F16, F16, 8, 16, 4, [1, 50, 299, 300, 301, 700], window=300)
    yield xqa_case("bf16_g2_p16_window_sinks", BF16, BF16, 2, 16, 8, MIXED, window=64, sinks=True)
