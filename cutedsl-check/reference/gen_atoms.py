#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Record NVIDIA CuTe DSL's answers about MMA and copy atoms, tiled MMAs,
tiled copies and tensor partitions, for seeded random cases.

    python gen_atoms.py [--seed N] [--scale N] OUT.tsv

Needs the `nvidia-cutlass-dsl` package (4.8.1 was used) on Linux; no GPU.
Each line of OUT.tsv is one question: the query, its arguments and what
CuTe DSL answered, as text (`ERR` where it refused, `CRASH` where the call
ended the Python process). `cutedsl-check atoms OUT.tsv` asks cute-rs the
same questions and compares.

Each case names the architecture it is traced for (`CUTE_DSL_ARCH`); the
cases of one architecture are answered in a child process started with that
architecture. The cases are drawn before CuTe DSL is asked anything, so a
seed always gives the same cases.

An MMA or copy operation is written `Name(arg,...)`: the CuTe DSL class
(`warp.MmaF16BF16Op`, `tcgen05.MmaI8Op`, `CopyUniversalOp`, ...) and its
arguments (types by their CuTe DSL names, enum members by their names).
A copy operation's first argument is the element type of the atom, and,
where CuTe DSL takes one, the second is `num_bits_per_copy`.
A tensor argument is a layout: the tensor is `make_tensor(0, layout)`,
whose iterator is an integer, so that a partition prints the offset of its
first element (`tensor<768o(4,2):(1,32)>`).
"""

import argparse
import os
import random
import subprocess
import sys

# ---------------------------------------------------------------- text

def text(x):
    if x is None:
        return "_"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, (tuple, list)):
        return "(" + ",".join(text(e) for e in x) + ")"
    return str(x).replace(" ", "")


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def flat(t):
    if isinstance(t, tuple):
        out = []
        for e in t:
            out += flat(e)
        return out
    return [t]


# An argument is ("S", spec) for an operation, ("L", (shape, stride)) for a
# layout or a tensor of it, ("P", (l0, l1, l2)) for a permutation (each a
# layout or None), ("I", n) or ("T", tuple).

def S(x):
    return ("S", x)


def L(shape, stride):
    return ("L", (shape, stride))


def P(x):
    return ("P", x)


def I(x):
    return ("I", x)


def T(x):
    return ("T", x)


def lay_text(lay):
    return text(lay[0]) + ":" + text(lay[1])


def arg_text(arg):
    kind, x = arg
    if kind == "S":
        return x[0] + "(" + ",".join(text(a) for a in x[1:]) + ")"
    if kind == "L":
        return lay_text(x)
    if kind == "LL":
        return ";".join(f"{e}:{d}" for e, d in x)
    if kind == "P":
        return "(" + ",".join("_" if p is None else lay_text(p) for p in x) + ")"
    return text(x)


# ------------------------------------------------------------- layouts

def col_major(shape):
    out, s = [], 1
    for e in shape:
        out.append(s)
        s *= e
    return tuple(out)


def row_major(shape):
    out, s = [], 1
    for e in reversed(shape):
        out.append(s)
        s *= e
    return tuple(reversed(out))


def padded(rng, shape):
    """Strides in a random order, with gaps."""
    order = list(range(len(shape)))
    rng.shuffle(order)
    strides = [0] * len(shape)
    s = 1
    for i in order:
        strides[i] = s
        s *= shape[i] + rng.choice((0, 0, 1, 4, 8))
    return tuple(strides)


def some_layout(rng, shape):
    r = rng.random()
    if r < 0.4:
        return (shape, col_major(shape))
    if r < 0.8:
        return (shape, row_major(shape))
    return (shape, padded(rng, shape))


def tensor_over(rng, tile):
    """A tensor of a few tiles of `tile` (a tuple of extents), sometimes with a batch mode."""
    shape = tuple(t * rng.choice((1, 1, 2, 2, 3)) for t in tile)
    if rng.random() < 0.2:
        shape = shape + (rng.choice((2, 3)),)
    return some_layout(rng, shape)


# ---------------------------------------------------------- MMA catalogue
# Each entry: (arch, spec, (M, N, K), threads per atom).

F16, BF16, F32, F64, TF32 = "Float16", "BFloat16", "Float32", "Float64", "TFloat32"
E4M3, E5M2 = "Float8E4M3FN", "Float8E5M2"
E2M3, E3M2, E2M1 = "Float6E2M3FN", "Float6E3M2FN", "Float4E2M1FN"
UE8M0 = "Float8E8M0FNU"
I8, U8, I32 = "Int8", "Uint8", "Int32"


def mma_specs_sm80(rng):
    out = []
    for ab, acc in ((F16, F16), (F16, F32), (BF16, F32)):
        for k in (8, 16):
            out.append(("sm_80", ("warp.MmaF16BF16Op", ab, acc, (16, 8, k)), (16, 8, k), 32))
    for k in (4, 8):
        out.append(("sm_80", ("warp.MmaTF32Op", (16, 8, k)), (16, 8, k), 32))
    for t in (F16, F32, F64):
        out.append(("sm_80", ("MmaUniversalOp", t), (1, 1, 1), 1))
    for ab in (E4M3, E5M2):
        for acc in (F16, F32):
            for k in (16, 32):
                out.append(("sm_89", ("warp.MmaFP8Op", ab, acc, (16, 8, k)), (16, 8, k), 32))
    return out


def mma_specs_sm120(rng):
    out = []
    out.append(("sm_120a", ("warp.MmaMXF4Op", E2M1, F32, UE8M0), (16, 8, 64), 32))
    for sf in (UE8M0, E4M3):
        out.append(("sm_120a", ("warp.MmaMXF4NVF4Op", E2M1, F32, sf), (16, 8, 64), 32))
    for ab in (E4M3, E5M2):
        out.append(("sm_120a", ("warp.MmaMXF8Op", ab, F32, UE8M0), (16, 8, 32), 32))
    types = (E4M3, E5M2, E2M3, E3M2, E2M1)
    for a in types:
        for b in types:
            out.append(("sm_120a", ("warp.MmaMXF8F6F4Op", a, b, F32, UE8M0), (16, 8, 32), 32))
    return out


def mma_specs_sm90(rng, n):
    out = []
    majors = ("K", "K", "MN")
    for _ in range(n):
        kind = rng.choice(("f16", "f16", "f8", "i8"))
        nn = rng.choice(range(8, 257, 8))
        src = rng.choice(("SMEM", "RMEM"))
        if kind == "f16":
            ab, acc = rng.choice(((F16, F16), (F16, F32), (BF16, F32)))
            am = "K" if src == "RMEM" and rng.random() < 0.9 else rng.choice(majors)
            spec = ("warpgroup.MmaF16BF16Op", ab, acc, (64, nn, 16), src, am, rng.choice(majors))
            k = 16
        elif kind == "f8":
            spec = ("warpgroup.MmaF8Op", rng.choice((E4M3, E5M2)), rng.choice((E4M3, E5M2)), rng.choice((F16, F32)),
                    (64, nn, 32), src, "K", "K")
            k = 32
        else:
            if nn > 32 and rng.random() < 0.8:
                nn = nn // 16 * 16 or 16
            spec = ("warpgroup.MmaI8Op", rng.choice((I8, U8)), rng.choice((I8, U8)), I32, (64, nn, 32), src, "K", "K")
            k = 32
        out.append(("sm_90a", spec, (64, nn, k), 128))
    return out


def umma_mn(rng, two):
    if two:
        m = rng.choice((128, 256))
        n = rng.choice(range(16 if m == 128 else 16, 257, 16))
    else:
        m = rng.choice((64, 128))
        n = rng.choice(range(8 if m == 64 else 16, 257, 8 if m == 64 else 16))
    return m, n


FP6_FP4 = (E2M3, E3M2, E2M1)


def mma_specs_sm100(rng, n):
    """tcgen05 MMAs. cute-rs does not model three corners, which are left out: A in tensor
    memory for a CTA pair or for FP6 and FP4 operands, and sparse MMAs of other than 128
    rows per CTA (or with A in tensor memory)."""
    out = []
    majors = ("K", "K", "MN")
    while len(out) < n:
        kind = rng.choice(("f16", "f16", "tf32", "i8", "f8f6f4", "mxf8f6f4", "mxf4", "mxf4nvf4", "sparse"))
        two = rng.random() < 0.3
        cta = "TWO" if two else "ONE"
        m, nn = umma_mn(rng, two)
        src = "TMEM" if not two and rng.random() < 0.25 else "SMEM"
        am = "K" if src == "TMEM" else rng.choice(majors)
        bm = rng.choice(majors)
        if kind == "f16":
            ab, acc = rng.choice(((F16, F16), (F16, F32), (BF16, F32)))
            spec, k = ("tcgen05.MmaF16BF16Op", ab, acc, (m, nn, 16), cta, src, am, bm), 16
        elif kind == "tf32":
            spec, k = ("tcgen05.MmaTF32Op", (m, nn, 8), cta, src, am, bm), 8
        elif kind == "i8":
            spec, k = ("tcgen05.MmaI8Op", rng.choice((I8, U8)), (m, nn, 32), cta, src, am, bm), 32
        elif kind in ("f8f6f4", "mxf8f6f4"):
            if kind == "mxf8f6f4" and m == 64:
                m = 128
            a = rng.choice((E4M3, E5M2) + FP6_FP4)
            b = rng.choice((E4M3, E5M2) + FP6_FP4)
            if a in FP6_FP4 or b in FP6_FP4:
                am, bm = "K", "K"
            if a in FP6_FP4:
                src = "SMEM"
            if kind == "f8f6f4":
                spec = ("tcgen05.MmaF8F6F4Op", a, b, rng.choice((F16, F32)), (m, nn, 32), cta, src, am, bm)
            else:
                spec = ("tcgen05.MmaMXF8F6F4Op", a, b, (m, nn, 32), cta, src, am, bm)
            k = 32
        elif kind == "mxf4":
            m = 128 if m == 64 else m
            spec, k = ("tcgen05.MmaMXF4Op", (m, nn, 64), cta, "SMEM"), 64
        elif kind == "mxf4nvf4":
            m = 128 if m == 64 else m
            spec, k = ("tcgen05.MmaMXF4NVF4Op", rng.choice((UE8M0, E4M3)), (m, nn, 64), cta, "SMEM"), 64
        else:
            m = 256 if two else 128
            ab, acc = rng.choice(((F16, F16), (F16, F32), (BF16, F32)))
            spec, k = ("tcgen05.MmaF16BF16SparseOp", ab, acc, (m, nn, 32), cta, "SMEM", am, bm, "TID"), 32
        out.append(("sm_100a", spec, (m, nn, k), 2 if two else 1))
    return out


def atom_layouts(threads, spec):
    if spec[0].startswith("tcgen05."):
        return [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0)), ((1, 2, 1), (0, 1, 0))]
    if threads == 1:
        return [((1, 1, 1), (0, 0, 0)), ((8, 8, 1), (1, 8, 0)), ((16, 8, 1), (1, 16, 0)), ((16, 16, 1), (16, 1, 0)),
                ((32, 4, 1), (1, 32, 0)), ((4, 8, 1), (8, 1, 0))]
    if threads == 32:
        return [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0)), ((1, 2, 1), (0, 1, 0)), ((2, 2, 1), (1, 2, 0)),
                ((2, 2, 1), (2, 1, 0)), ((4, 1, 1), (1, 0, 0)), ((1, 4, 1), (0, 1, 0)), ((4, 2, 1), (1, 4, 0)),
                ((2, 4, 1), (1, 2, 0)), ((2, 4, 1), (4, 1, 0)), ((1, 1, 2), (0, 0, 1)), ((2, 1, 2), (1, 0, 2))]
    return [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0)), ((1, 2, 1), (0, 1, 0)), ((4, 1, 1), (1, 0, 0)),
            ((2, 2, 1), (1, 2, 0))]


def permutation(rng, natural):
    """None per mode, or a tile of the natural size or twice it, plain or interleaved."""
    if rng.random() < 0.55:
        return (None, None, None)
    out = []
    for n in natural:
        r = rng.random()
        if r < 0.4:
            out.append(None)
        elif r < 0.7:
            out.append((n, 1))
        elif r < 0.85:
            out.append((2 * n, 1))
        else:
            out.append(((n, 2), (2, 1)))
    return tuple(out)


def tile_sizes(mnk, al, perm):
    natural = tuple(mnk[i] * al[0][i] for i in range(3))
    return tuple(natural[i] if perm[i] is None else prod(flat(perm[i][0])) for i in range(3))


def lay(x):
    """A layout argument from a (shape, stride) pair."""
    return ("L", x)


# ---------------------------------------------------------------- cases

def mma_atom_cases(specs):
    out = []
    for arch, spec, _, _ in specs:
        for q in ("shape_mnk", "thr_id", "tv_A", "tv_B", "tv_C"):
            out.append((arch, "mma_atom." + q, [S(spec)]))
    return out


def operand_types(spec):
    """The element types of A, B and C of an MMA operation."""
    name, a = spec[0], spec[1:]
    if name.endswith("TF32Op"):
        return (TF32, TF32, F32)
    if name == "MmaUniversalOp":
        return (a[0], a[0], a[0])
    if name in ("warp.MmaMXF8F6F4Op", "warpgroup.MmaF8Op", "warpgroup.MmaI8Op", "tcgen05.MmaF8F6F4Op"):
        return (a[0], a[1], a[2])
    if name == "tcgen05.MmaI8Op":
        return (a[0], a[0], I32)
    if name == "tcgen05.MmaMXF8F6F4Op":
        return (a[0], a[1], F32)
    if name in ("tcgen05.MmaMXF4Op", "tcgen05.MmaMXF4NVF4Op"):
        return (E2M1, E2M1, F32)
    return (a[0], a[0], a[1])


def has_register_a(spec):
    name = spec[0]
    if name.startswith("warp.") or name == "MmaUniversalOp":
        return True
    return "RMEM" in spec or "TMEM" in spec


def tiled_mma_cases(rng, specs, count):
    out = []
    for _ in range(count):
        arch, spec, mnk, threads = rng.choice(specs)
        al = rng.choice(atom_layouts(threads, spec))
        natural = tuple(mnk[i] * al[0][i] for i in range(3))
        perm = permutation(rng, natural)
        tiles = tile_sizes(mnk, al, perm)
        size = threads * prod(al[0])
        base = [S(spec), lay(al), P(perm)]
        for q in ("thr_layout_vmnk", "tile_size", "size", "tv_A", "tv_B", "tv_C"):
            out.append((arch, "tiled_mma." + q, list(base)))
        for which, (i, j) in (("A", (0, 2)), ("B", (1, 2)), ("C", (0, 1))):
            shape = tuple(t * rng.choice((1, 2)) for t in (tiles[i], tiles[j]))
            out.append((arch, "tiled_mma.partition_shape_" + which, base + [T(shape)]))
            t = tensor_over(rng, (tiles[i], tiles[j]))
            out.append((arch, "tiled_mma.partition_" + which, base + [lay(t), I(rng.randrange(size))]))
            if which == "C" or (which == "A" and has_register_a(spec)) or (which == "B" and spec[0].startswith("warp.")) \
                    or (which == "B" and spec[0] == "MmaUniversalOp"):
                t = tensor_over(rng, (tiles[i], tiles[j]))
                out.append((arch, "tiled_mma.fragment_" + which, base + [lay(t), I(rng.randrange(size))]))
    return out


# Copy atoms: (arch, spec, values a thread holds in the atom's reference layout, threads).

def copy_specs(rng):
    out = []
    widths = {F16: 16, BF16: 16, F32: 32, F64: 64, I8: 8, U8: 8, I32: 32, E4M3: 8}
    for t in (F16, BF16, F32, F64, I8, I32):
        for bits in (8, 16, 32, 64, 128):
            if bits >= widths[t]:
                out.append(("sm_80", ("CopyUniversalOp", t, bits), bits // widths[t], 1))
    for t in (F16, F32, I8):
        for bits in (32, 64, 128):
            if bits >= widths[t]:
                out.append(("sm_80", ("cpasync.CopyG2SOp", t, bits, "ALWAYS"), bits // widths[t], 1))
        out.append(("sm_80", ("cpasync.CopyG2SOp", t, 128, "GLOBAL"), 128 // widths[t], 1))
    for name in ("CopyG2ROp", "CopyR2GOp", "CopyS2ROp", "CopyR2SOp"):
        for t in (F16, F32, I8):
            for bits in (32, 128):
                if bits >= widths[t]:
                    out.append(("sm_80", (name, t, bits), bits // widths[t], 1))
    for t in (F16, BF16):
        for trans in (False, True):
            for num in (1, 2, 4):
                out.append(("sm_80", ("warp.LdMatrix8x8x16bOp", t, trans, num), 2 * num, 32))
                out.append(("sm_90a", ("warp.StMatrix8x8x16bOp", t, trans, num), 2 * num, 32))
    for t in (E4M3, I8, U8):
        for num in (1, 2, 4):
            for unpack in (None, 4, 6):
                out.append(("sm_100a", ("warp.LdMatrix8x16x8bOp", t, num, unpack), 4 * num, 32))
        for num in (1, 2):
            for unpack in (None, 4, 6):
                out.append(("sm_100a", ("warp.LdMatrix16x16x8bOp", t, True, num, unpack), 8 * num, 32))
        for num in (2, 4):
            for unpack in (None, 4, 6):
                out.append(("sm_100a", ("warp.LdMatrix16x8x8bOp", t, True, num, unpack), 4 * num, 32))
            out.append(("sm_100a", ("warp.StMatrix16x8x8bOp", t, True, num), 4 * num, 32))
    reps = ("x1", "x2", "x4", "x8", "x16", "x32", "x64", "x128")
    for shape in ("32x32b", "16x64b", "16x128b", "16x256b", "16x32bx2"):
        for rep in reps:
            # Packing moves 16-bit halves: cute-rs takes it only for elements of 16 bits or fewer.
            for ty, pack in ((F32, "NONE"), (F16, "NONE"), (F16, "PACK_16b_IN_32b")):
                out.append(("sm_100a", ("tcgen05.Ld" + shape + "Op", ty, rep, pack), 0, 32))
            for ty, unpack in ((F32, "NONE"), (F16, "NONE"), (F16, "UNPACK_32b_IN_16b")):
                out.append(("sm_100a", ("tcgen05.St" + shape + "Op", ty, rep, unpack), 0, 32))
    return out


def copy_atom_cases(specs):
    out = []
    for arch, spec, _, _ in specs:
        for q in ("thr_id", "src_tv", "dst_tv"):
            out.append((arch, "copy_atom." + q, [S(spec)]))
    return out


def thr_layout(rng, threads):
    if threads == 32:
        shapes = [(32, 1), (8, 4), (4, 8), (16, 2), (2, 16), (32, 2), (16, 4), (8, 8)]
    else:
        shapes = [(8, 4), (4, 8), (16, 8), (8, 16), (32, 4), (32, 1), (1, 32), (16, 16), (8, 8), (4, 4)]
    shape = rng.choice(shapes)
    return (shape, col_major(shape) if rng.random() < 0.5 else row_major(shape))


def val_layout(rng, v):
    v2 = v * rng.choice((1, 1, 2))
    shape = rng.choice(((v2, 1), (1, v2), (v2, 2), (2, v2), (v, 1), (1, v)))
    return (shape, col_major(shape) if rng.random() < 0.7 else row_major(shape))


def tiled_copy_cases(rng, specs, count):
    out = []
    usable = [s for s in specs if s[2] > 0]
    for _ in range(count):
        arch, spec, v, threads = rng.choice(usable)
        thr = thr_layout(rng, threads)
        val = val_layout(rng, v)
        tiler = (thr[0][0] * val[0][0], thr[0][1] * val[0][1])
        nthr = prod(thr[0])
        base = [S(spec), lay(thr), lay(val)]
        for q in ("layout_tv", "tiler", "src_tv_tiled", "dst_tv_tiled", "size"):
            out.append((arch, "tiled_copy_tv." + q, list(base)))
        for q in ("partition_S", "partition_D"):
            out.append((arch, "tiled_copy_tv." + q, base + [lay(tensor_over(rng, tiler)), I(rng.randrange(nthr))]))
        frag = (prod(val[0]), rng.choice((1, 2)), rng.choice((1, 2)))
        out.append((arch, "tiled_copy_tv.retile", base + [lay((frag, col_major(frag)))]))
    return out


def mma_copy_cases(rng, mma_specs, cspecs, count):
    """make_tiled_copy_A/B/C and make_tiled_copy_C_atom of copy atoms over tiled MMAs."""
    out = []
    warp_mmas = [s for s in mma_specs if s[3] == 32 and s[1][0] in ("warp.MmaF16BF16Op", "warp.MmaTF32Op", "warp.MmaFP8Op")]
    for _ in range(count):
        arch, spec, mnk, threads = rng.choice(warp_mmas)
        al = rng.choice(atom_layouts(threads, spec))
        natural = tuple(mnk[i] * al[0][i] for i in range(3))
        perm = permutation(rng, natural)
        tiles = tile_sizes(mnk, al, perm)
        size = threads * prod(al[0])
        which = rng.choice(("A", "B", "C", "C", "C_atom"))
        ab = spec[1] if spec[0] != "warp.MmaTF32Op" else F32
        acc = spec[2] if spec[0] != "warp.MmaTF32Op" else F32
        ty = acc if which.startswith("C") else ab
        widths = {F16: 16, BF16: 16, F32: 32, E4M3: 8, E5M2: 8}
        w = widths[ty]
        choices = [("sm_80", ("CopyUniversalOp", ty, b)) for b in (16, 32, 64, 128) if b >= w]
        if which in ("A", "B") and w == 16:
            choices += [("sm_80", ("warp.LdMatrix8x8x16bOp", ty, which == "B" and rng.random() < 0.3, n)) for n in (1, 2, 4)]
        if which in ("A", "B") and w == 8:
            choices += [("sm_100a", ("warp.LdMatrix8x16x8bOp", ty, n, None)) for n in (1, 2, 4)]
        if which.startswith("C"):
            for t2 in (F16, BF16):
                choices += [("sm_90a", ("warp.StMatrix8x8x16bOp", t2, False, n)) for n in (1, 2, 4)]
            choices += [("sm_80", ("CopyUniversalOp", F16, 32))]
        carch, cspec = rng.choice(choices)
        order = ["sm_80", "sm_89", "sm_90a", "sm_100a"]
        use = max(arch, carch, key=order.index)
        base = [S(cspec), S(spec), lay(al), P(perm)]
        name = "tiled_copy_" + which
        out.append((use, name + ".layout_tv", list(base)))
        out.append((use, name + ".tiler", list(base)))
        i, j = {"A": (0, 2), "B": (1, 2)}.get(which, (0, 1))
        for q in ("partition_S", "partition_D"):
            out.append((use, name + "." + q, base + [lay(tensor_over(rng, (tiles[i], tiles[j]))), I(rng.randrange(size))]))
    return out


def cotiled_cases(rng, specs, count):
    out = []
    usable = [s for s in specs if s[3] == 1 and s[1][0] == "CopyUniversalOp"]
    for _ in range(count):
        arch, spec, v, _ = rng.choice(usable)
        t = rng.choice((32, 64, 128))
        m = rng.choice((8, 16, 32, 64, 128))
        n = rng.choice((8, 16, 32, 64))
        data = some_layout(rng, (m, n))
        if rng.random() < 0.5:
            atv = ((t, v), (v, 1))
        else:
            atv = ((t, v), (1, t))
        base = [S(spec), lay(atv), lay(data)]
        out.append((arch, "cotiled_copy.layout_tv", list(base)))
        out.append((arch, "cotiled_copy.tiler", list(base)))
    return out


def retarget_cases(rng, specs, count):
    """make_tiled_copy_S/D: another atom over a tiled copy's layout."""
    out = []
    univ = [s for s in specs if s[1][0] == "CopyUniversalOp" and s[1][1] == F16]
    ldsm = [s for s in specs if s[1][0] in ("warp.LdMatrix8x8x16bOp", "CopyUniversalOp", "cpasync.CopyG2SOp") and s[1][1] == F16]
    for _ in range(count):
        arch, spec, v, threads = rng.choice(univ)
        thr = thr_layout(rng, 32)
        val = val_layout(rng, v)
        arch2, spec2, _, _ = rng.choice(ldsm)
        which = rng.choice(("S", "D"))
        base = [S(spec2), S(spec), lay(thr), lay(val)]
        for q in ("layout_tv", "tiler"):
            out.append((max(arch, arch2), "tiled_copy_" + which + "." + q, list(base)))
    return out


def rand_shape(rng, rank):
    return tuple(rng.choice((2, 3, 4, 4, 6, 8, 8, 16)) for _ in range(rank))


def partition_cases(rng, count):
    out = []
    for _ in range(count):
        r = rng.random()
        rank = rng.choice((2, 2, 3))
        tile = rand_shape(rng, rank)
        mult = tuple(rng.choice((1, 2, 3)) for _ in tile)
        shape = tuple(t * m for t, m in zip(tile, mult))
        t = some_layout(rng, shape)
        if r < 0.3:
            coord = tuple(None if rng.random() < 0.3 else rng.randrange(m) for m in mult)
            out.append(("sm_80", "local_tile", [lay(t), T(tile), T(coord)]))
        elif r < 0.4:
            proj = tuple(1 if rng.random() < 0.6 else None for _ in tile)
            if all(p is None for p in proj):
                proj = (1,) + proj[1:]
            coord = tuple(None if rng.random() < 0.3 else rng.randrange(m) for m in mult)
            out.append(("sm_80", "local_tile_proj", [lay(t), T(tile), T(coord), T(proj)]))
        elif r < 0.65:
            thr = some_layout(rng, tile)
            out.append(("sm_80", "local_partition", [lay(t), lay(thr), I(rng.randrange(prod(tile)))]))
        elif r < 0.7:
            thr = some_layout(rng, tile)
            proj = tuple(1 if rng.random() < 0.6 else None for _ in tile)
            if all(p is None for p in proj):
                proj = (1,) + proj[1:]
            out.append(("sm_80", "local_partition_proj", [lay(t), lay(thr), I(rng.randrange(prod(tile))), T(proj)]))
        elif r < 0.9:
            op = rng.choice(("logical_divide", "zipped_divide", "tiled_divide", "flat_divide"))
            # A tile of stride 2 needs at least two tiles' extent in its mode.
            tilers = tuple((e, rng.choice((1, 1, 2)) if m >= 2 else 1) for e, m in zip(tile, mult))
            out.append(("sm_80", "tensor." + op, [lay(t), ("LL", tilers)]))
        else:
            coord = tuple(None if rng.random() < 0.5 else rng.randrange(e) for e in shape)
            out.append(("sm_80", "tensor.slice", [lay(t), T(coord)]))
    return out


def draw(seed, scale):
    """Every case, in order; the same for a seed and scale."""
    cases = []
    rng = random.Random(seed * 1000 + 1)
    specs = mma_specs_sm80(rng) + mma_specs_sm90(rng, 12 * scale) + mma_specs_sm100(rng, 24 * scale) + mma_specs_sm120(rng)
    cases += mma_atom_cases(specs)
    rng = random.Random(seed * 1000 + 2)
    sm80 = [s for s in specs if s[0] in ("sm_80", "sm_89")]
    sm90 = [s for s in specs if s[0] == "sm_90a"]
    sm100 = [s for s in specs if s[0] == "sm_100a"]
    sm120 = [s for s in specs if s[0] == "sm_120a"]
    cases += tiled_mma_cases(rng, sm80, 16 * scale)
    cases += tiled_mma_cases(rng, sm90, 8 * scale)
    cases += tiled_mma_cases(rng, sm100, 10 * scale)
    cases += tiled_mma_cases(rng, sm120, 3 * scale)
    rng = random.Random(seed * 1000 + 3)
    cspecs = copy_specs(rng)
    cases += copy_atom_cases(cspecs)
    rng = random.Random(seed * 1000 + 4)
    cases += tiled_copy_cases(rng, cspecs, 20 * scale)
    rng = random.Random(seed * 1000 + 5)
    cases += mma_copy_cases(rng, specs, cspecs, 20 * scale)
    rng = random.Random(seed * 1000 + 6)
    cases += cotiled_cases(rng, cspecs, 5 * scale)
    rng = random.Random(seed * 1000 + 7)
    cases += retarget_cases(rng, cspecs, 5 * scale)
    rng = random.Random(seed * 1000 + 8)
    cases += partition_cases(rng, 60 * scale)
    return cases


# ------------------------------------------------------------ CuTe DSL

class Unknown(Exception):
    """A query or operation the generator does not know."""


def dsl_answers(cases, arch, part):
    """Answer the cases of `arch` (every other line is left empty), in this
    process, which was started with CUTE_DSL_ARCH set to `arch`."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute import nvgpu
    from cutlass.cute.nvgpu import warp, warpgroup, tcgen05, cpasync

    def dt(name):
        return getattr(cutlass, name)

    def mk(x):
        return cute.make_layout(x[0], stride=x[1])

    def major(m):
        return getattr(nvgpu.OperandMajorMode, m)

    def mma_op(spec):
        name, a = spec[0], spec[1:]
        if name == "warp.MmaF16BF16Op":
            return warp.MmaF16BF16Op(dt(a[0]), dt(a[1]), a[2])
        if name == "warp.MmaTF32Op":
            return warp.MmaTF32Op(a[0])
        if name == "warp.MmaFP8Op":
            return warp.MmaFP8Op(dt(a[0]), dt(a[1]), a[2])
        if name == "MmaUniversalOp":
            return nvgpu.MmaUniversalOp(dt(a[0]))
        if name in ("warp.MmaMXF4Op", "warp.MmaMXF4NVF4Op", "warp.MmaMXF8Op"):
            return getattr(warp, name[5:])(dt(a[0]), dt(a[1]), dt(a[2]))
        if name == "warp.MmaMXF8F6F4Op":
            return warp.MmaMXF8F6F4Op(dt(a[0]), dt(a[1]), dt(a[2]), dt(a[3]))
        if name == "warpgroup.MmaF16BF16Op":
            return warpgroup.MmaF16BF16Op(dt(a[0]), dt(a[1]), a[2], getattr(warpgroup.OperandSource, a[3]),
                                          major(a[4]), major(a[5]))
        if name in ("warpgroup.MmaF8Op", "warpgroup.MmaI8Op"):
            return getattr(warpgroup, name[10:])(dt(a[0]), dt(a[1]), dt(a[2]), a[3],
                                                 getattr(warpgroup.OperandSource, a[4]), major(a[5]), major(a[6]))
        cta = lambda s: getattr(tcgen05.CtaGroup, s)
        src = lambda s: getattr(tcgen05.OperandSource, s)
        if name == "tcgen05.MmaF16BF16Op":
            return tcgen05.MmaF16BF16Op(dt(a[0]), dt(a[1]), a[2], cta(a[3]), src(a[4]), major(a[5]), major(a[6]))
        if name == "tcgen05.MmaTF32Op":
            return tcgen05.MmaTF32Op(a[0], cta(a[1]), src(a[2]), major(a[3]), major(a[4]))
        if name == "tcgen05.MmaI8Op":
            return tcgen05.MmaI8Op(dt(a[0]), a[1], cta(a[2]), src(a[3]), major(a[4]), major(a[5]))
        if name == "tcgen05.MmaF8F6F4Op":
            return tcgen05.MmaF8F6F4Op(dt(a[0]), dt(a[1]), dt(a[2]), a[3], cta(a[4]), src(a[5]), major(a[6]), major(a[7]))
        if name == "tcgen05.MmaMXF8F6F4Op":
            return tcgen05.MmaMXF8F6F4Op(dt(a[0]), dt(a[1]), a[2], cta(a[3]), src(a[4]), major(a[5]), major(a[6]))
        if name == "tcgen05.MmaMXF4Op":
            return tcgen05.MmaMXF4Op(a[0], cta(a[1]), src(a[2]))
        if name == "tcgen05.MmaMXF4NVF4Op":
            return tcgen05.MmaMXF4NVF4Op(dt(a[0]), a[1], cta(a[2]), src(a[3]))
        if name == "tcgen05.MmaF16BF16SparseOp":
            return tcgen05.MmaF16BF16SparseOp(dt(a[0]), dt(a[1]), a[2], cta(a[3]), src(a[4]), major(a[5]), major(a[6]),
                                              getattr(warp.SparseMetadataFormat, a[7]))
        raise Unknown(name)

    def copy_atom(spec):
        name, a = spec[0], spec[1:]
        ty = dt(a[0])
        if name == "CopyUniversalOp":
            return cute.make_copy_atom(nvgpu.CopyUniversalOp(), ty, num_bits_per_copy=a[1])
        if name == "cpasync.CopyG2SOp":
            return cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=getattr(cpasync.LoadCacheMode, a[2])), ty,
                                       num_bits_per_copy=a[1])
        if name in ("CopyG2ROp", "CopyR2GOp", "CopyS2ROp", "CopyR2SOp"):
            return cute.make_copy_atom(getattr(nvgpu, name)(), ty, num_bits_per_copy=a[1])
        if name in ("warp.LdMatrix8x8x16bOp", "warp.StMatrix8x8x16bOp"):
            return cute.make_copy_atom(getattr(warp, name[5:])(a[1], a[2]), ty)
        if name == "warp.LdMatrix8x16x8bOp":
            return cute.make_copy_atom(warp.LdMatrix8x16x8bOp(False, a[1], a[2]), ty)
        if name in ("warp.LdMatrix16x16x8bOp", "warp.LdMatrix16x8x8bOp"):
            return cute.make_copy_atom(getattr(warp, name[5:])(a[1], a[2], a[3]), ty)
        if name == "warp.StMatrix16x8x8bOp":
            return cute.make_copy_atom(warp.StMatrix16x8x8bOp(a[1], a[2]), ty)
        if name.startswith("tcgen05.Ld"):
            return cute.make_copy_atom(getattr(tcgen05, name[8:])(getattr(tcgen05.Repetition, a[1]),
                                                                  getattr(tcgen05.Pack, a[2])), ty)
        if name.startswith("tcgen05.St"):
            return cute.make_copy_atom(getattr(tcgen05, name[8:])(getattr(tcgen05.Repetition, a[1]),
                                                                  getattr(tcgen05.Unpack, a[2])), ty)
        raise Unknown(name)

    def perm(p):
        if all(x is None for x in p):
            return None
        return tuple(None if x is None else mk(x) for x in p)

    def tiled_mma(spec, al, p):
        return cute.make_tiled_mma(mma_op(spec), mk(al), permutation_mnk=perm(p))

    def counting(x):
        return cute.make_tensor(0, mk(x))

    def memory(x, ty):
        # A global-memory tensor of the operand's type, for make_fragment_A/B/C.
        return cute.make_tensor(cute.make_ptr(dt(ty), 0, cute.AddressSpace.gmem, assumed_align=16), mk(x))

    def answer(op, args):
        v = [a[1] for a in args]
        if op.startswith("mma_atom."):
            atom = cute.make_mma_atom(mma_op(v[0]))
            q = op[9:]
            return {"shape_mnk": lambda: atom.shape_mnk, "thr_id": lambda: atom.thr_id, "tv_A": lambda: atom.tv_layout_A,
                    "tv_B": lambda: atom.tv_layout_B, "tv_C": lambda: atom.tv_layout_C}[q]()
        if op.startswith("tiled_mma."):
            tm = tiled_mma(v[0], v[1], v[2])
            q = op[10:]
            if q == "thr_layout_vmnk":
                return tm.thr_layout_vmnk
            if q == "tile_size":
                return tuple(tm.get_tile_size(i) for i in range(3))
            if q == "size":
                return tm.size
            if q in ("tv_A", "tv_B", "tv_C"):
                return getattr(tm, "tv_layout_" + q[3] + "_tiled")
            if q.startswith("partition_shape_"):
                return getattr(tm, "partition_shape_" + q[-1])(v[3])
            if q.startswith("partition_"):
                return getattr(tm.get_slice(v[4]), "partition_" + q[-1])(counting(v[3]))
            if q.startswith("fragment_"):
                w = q[-1]
                ty = operand_types(v[0])["ABC".index(w)]
                p = getattr(tm.get_slice(v[4]), "partition_" + w)(memory(v[3], ty))
                return getattr(tm, "make_fragment_" + w)(p).layout
            raise Unknown(op)
        if op.startswith("copy_atom."):
            a = copy_atom(v[0])
            return {"thr_id": lambda: a.thr_id, "src_tv": lambda: a.layout_src_tv, "dst_tv": lambda: a.layout_dst_tv}[op[10:]]()
        if op.startswith("tiled_copy_tv."):
            tc = cute.make_tiled_copy_tv(copy_atom(v[0]), mk(v[1]), mk(v[2]))
            return tiled_copy_query(tc, op[14:], v[3:])
        if op.split(".")[0] in ("tiled_copy_A", "tiled_copy_B", "tiled_copy_C", "tiled_copy_C_atom"):
            name, q = op.split(".")
            tm = tiled_mma(v[1], v[2], v[3])
            tc = getattr(cute, "make_" + name)(copy_atom(v[0]), tm)
            return tiled_copy_query(tc, q, v[4:])
        if op.startswith("cotiled_copy."):
            tc = cute.make_cotiled_copy(copy_atom(v[0]), mk(v[1]), mk(v[2]))
            return tiled_copy_query(tc, op[13:], v[3:])
        if op.split(".")[0] in ("tiled_copy_S", "tiled_copy_D"):
            name, q = op.split(".")
            inner = cute.make_tiled_copy_tv(copy_atom(v[1]), mk(v[2]), mk(v[3]))
            tc = getattr(cute, "make_" + name)(copy_atom(v[0]), inner)
            return tiled_copy_query(tc, q, v[4:])
        if op == "local_tile":
            return cute.local_tile(counting(v[0]), v[1], v[2])
        if op == "local_tile_proj":
            return cute.local_tile(counting(v[0]), v[1], v[2], proj=v[3])
        if op == "local_partition":
            return cute.local_partition(counting(v[0]), mk(v[1]), cutlass.Int32(v[2]))
        if op == "local_partition_proj":
            return cute.local_partition(counting(v[0]), mk(v[1]), cutlass.Int32(v[2]), proj=v[3])
        if op.startswith("tensor.") and op.endswith("_divide"):
            tiler = tuple(cute.make_layout(e, stride=s) for e, s in v[1])
            return getattr(cute, op[7:])(counting(v[0]), tiler)
        if op == "tensor.slice":
            return counting(v[0])[v[1]]
        raise Unknown(op)

    def tiled_copy_query(tc, q, rest):
        if q == "layout_tv":
            return tc.layout_tv_tiled
        if q == "tiler":
            return tc.tiler_mn
        if q == "src_tv_tiled":
            return tc.layout_src_tv_tiled
        if q == "dst_tv_tiled":
            return tc.layout_dst_tv_tiled
        if q == "size":
            return tc.size
        if q == "partition_S":
            return tc.get_slice(rest[1]).partition_S(counting(rest[0]))
        if q == "partition_D":
            return tc.get_slice(rest[1]).partition_D(counting(rest[0]))
        if q == "retile":
            return tc.retile(cute.make_rmem_tensor(mk(rest[0]), cutlass.Float32)).layout
        raise Unknown(q)

    mine = [n for n, c in enumerate(cases) if c[0] == arch]
    state = {"start": 0}

    class Recorded(Exception):
        pass

    def answer_all():
        with open(part, "a", buffering=1) as f:
            for n in mine[state["start"]:]:
                _, op, args = cases[n]
                try:
                    result = text(answer(op, args))
                except Unknown:
                    raise
                except Exception:
                    result = "ERR"
                f.write(f"{n}\t{result}\n")

    @cute.jit
    def traced():
        # CuTe DSL's atoms and layouts only answer while a function is being traced.
        answer_all()
        raise Recorded()

    def answered():
        with open(part) as f:
            return sum(1 for _ in f)

    open(part, "w").close()
    while (done := answered()) < len(mine):
        state["start"] = done
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                traced()
            except Recorded:
                pass
            except Unknown as e:
                print(f"unknown operation {e}", file=sys.stderr)
                code = 3
            except BaseException:
                pass
            os._exit(code)
        _, status = os.waitpid(pid, 0)
        if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 3:
            raise SystemExit("an operation the generator does not know")
        now = answered()
        if now < len(mine):
            if not os.WIFSIGNALED(status):
                # An exception that escaped the trace (CuTe DSL can raise
                # while the traced function unwinds): answer the case as ERR.
                pass
            with open(part, "a") as f:
                f.write(f"{mine[now]}\t{'CRASH' if os.WIFSIGNALED(status) else 'ERR'}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--answer", help=argparse.SUPPRESS)
    opts = ap.parse_args()
    cases = draw(opts.seed, opts.scale)

    if opts.answer:
        dsl_answers(cases, opts.answer, opts.out)
        return

    results = {}
    for arch in sorted({c[0] for c in cases}):
        part = f"{opts.out}.{arch}.part"
        env = dict(os.environ, CUTE_DSL_ARCH=arch)
        cmd = [sys.executable, os.path.abspath(__file__), part, "--seed", str(opts.seed), "--scale", str(opts.scale),
               "--answer", arch]
        subprocess.run(cmd, env=env, check=True)
        with open(part) as f:
            for line in f:
                n, r = line.rstrip("\n").split("\t", 1)
                results[int(n)] = r
        os.remove(part)

    import cutlass

    with open(opts.out, "w") as f:
        f.write(f"# cute-dsl {getattr(cutlass, '__version__', '?')} seed {opts.seed} scale {opts.scale}\n")
        for n, (arch, op, args) in enumerate(cases):
            f.write("\t".join([op, arch] + [arg_text(a) for a in args] + ["=>", results[n]]) + "\n")
    refused = sum(r == "ERR" for r in results.values())
    crashed = sum(r == "CRASH" for r in results.values())
    print(f"{len(cases)} cases: CuTe DSL refused {refused} and crashed on {crashed} -> {opts.out}")


if __name__ == "__main__":
    main()
