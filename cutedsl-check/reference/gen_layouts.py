#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Record NVIDIA CuTe DSL's answers to seeded random layout-algebra calls.

    python gen_layouts.py [--seed N] [--per-op N] OUT.tsv

Needs the `nvidia-cutlass-dsl` package (4.8.1 was used) on Linux; no GPU.
Each line of OUT.tsv is one call: the operation, its arguments and what
CuTe DSL returned, as text (`ERR` where it refused, `CRASH` where the call
ended the Python process). `cutedsl-check layouts
OUT.tsv` then makes the same calls in cute-rs and compares.

The cases are drawn before CuTe DSL is asked anything, so a seed always
gives the same calls.
"""

import argparse
import os
import random

import cutlass.cute as cute

EXTENTS = [1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 8]
STRIDES = [0, 1, 1, 2, 2, 3, 4, 6, 8, 12, 16]
SMALL = (1, 2, 3, 4, 6, 8)


# ---------------------------------------------------------------- trees

def flat(t):
    if isinstance(t, tuple):
        out = []
        for e in t:
            out += flat(e)
        return out
    return [t]


def like(t, values):
    """The tree `t` with its leaves replaced, in order, from `values`."""
    if isinstance(t, tuple):
        return tuple(like(e, values) for e in t)
    return next(values)


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


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


# ------------------------------------------------------- random layouts

def rand_shape(rng, ranks=(1, 2, 2, 3, 3, 4), nest=0.2, bare=0.5):
    r = rng.choice(ranks)
    if r == 1 and rng.random() < bare:
        return rng.choice(EXTENTS)
    return tuple(
        tuple(rng.choice(EXTENTS) for _ in range(rng.choice((2, 2, 3)))) if rng.random() < nest else rng.choice(EXTENTS)
        for _ in range(r)
    )


def compact(rng, shape, pad=False, scale=1):
    """Strides that visit the leaves in a random order; with `pad`, with gaps."""
    ext = flat(shape)
    order = list(range(len(ext)))
    rng.shuffle(order)
    strides = [0] * len(ext)
    s = scale
    for i in order:
        strides[i] = s
        e = ext[i]
        if pad:
            e = rng.choice((e, e, e + 1, 2 * e))
        s *= e
    return like(shape, iter(strides))


def loose(rng, shape):
    return like(shape, iter([rng.choice(STRIDES) for _ in flat(shape)]))


def layout(rng, kind=None, **kw):
    """A (shape, stride) pair: a bijection, an injection with gaps, or anything."""
    kind = kind or rng.choice(("bij", "bij", "inj", "inj", "any"))
    shape = rand_shape(rng, **kw)
    if kind == "bij":
        stride = compact(rng, shape)
    elif kind == "inj":
        stride = compact(rng, shape, pad=rng.random() < 0.7, scale=rng.choice((1, 1, 2, 3, 4)))
    else:
        stride = loose(rng, shape)
    return (shape, stride)


def size(lay):
    return prod(flat(lay[0]))


def cosize(lay):
    return 1 + sum((e - 1) * s for e, s in zip(flat(lay[0]), flat(lay[1])))


def divisors(n):
    return [d for d in SMALL if n % d == 0]


# ---------------------------------------------------------------- cases
# An argument is ("L", (shape, stride)), ("I", int), ("T", tuple) or
# ("LL", [layouts]).

def L(x):
    return ("L", x)


def I(x):
    return ("I", x)


def T(x):
    return ("T", x)


def one_layout(kinds=None, **kw):
    def gen(rng):
        return [L(layout(rng, rng.choice(kinds) if kinds else None, **kw))]
    return gen


def gen_idx2crd(rng):
    shape = rand_shape(rng, bare=0.2)
    n = prod(flat(shape))
    i = rng.randrange(n) if rng.random() < 0.9 else rng.randrange(n, 2 * n + 1)
    return [I(i), T(shape)]


def gen_crd2idx(rng):
    lay = layout(rng)
    crd = like(lay[0], iter([rng.randrange(e) for e in flat(lay[0])]))
    return [T(crd), L(lay)]


def gen_composition(rng):
    a = layout(rng)
    b = layout(rng)
    if rng.random() < 0.75:
        for _ in range(50):
            if cosize(b) <= size(a):
                break
            b = layout(rng)
    return [L(a), L(b)]


def gen_complement(rng):
    a = layout(rng, rng.choice(("bij", "inj", "inj", "any")))
    c = cosize(a)
    m = rng.choice((c, c, size(a) * rng.choice((1, 2, 3)), c + rng.randrange(2 * c + 1), rng.randrange(1, 2 * c + 2), (c + 7) // 8 * 8))
    return [L(a), I(m)]


def gen_logical_divide(rng):
    a = layout(rng)
    if rng.random() < 0.75:
        t = rng.choice(divisors(size(a)))
        tiler = (t, rng.choice((1, 1, 1, 2, 3, 4)))
    else:
        tiler = layout(rng)
    return [L(a), L(tiler)]


def gen_tuple_divide(rng):
    a = layout(rng, ranks=(2, 3, 3, 4), bare=0.0)
    modes = a[0]
    tilers = []
    for i in range(rng.randint(1, len(modes))):
        n = prod(flat(modes[i]))
        t = rng.choice(divisors(n)) if rng.random() < 0.85 else rng.choice((2, 3, 5))
        tilers.append((t, rng.choice((1, 1, 1, 2))))
    return [L(a), ("LL", tilers)]


def gen_product(rng):
    return [L(layout(rng, rng.choice(("bij", "inj", "any")))), L(layout(rng, ranks=(1, 2, 2, 3)))]


def gen_tile_to_shape(rng):
    block = layout(rng, "bij", ranks=(2, 2, 3), nest=0.0, bare=0.0)
    if rng.random() < 0.8:
        target = tuple(e * rng.choice((1, 2, 3, 4)) for e in block[0])
    else:
        target = tuple(rng.choice(EXTENTS) for _ in block[0])
    order = list(range(len(block[0])))
    rng.shuffle(order)
    return [L(block), T(target), T(tuple(order))]


def gen_layout_tv(rng):
    return [L(layout(rng, rng.choice(("bij", "inj")), ranks=(1, 2, 2))), L(layout(rng, "bij", ranks=(1, 1, 2)))]


def gen_recast(rng):
    bits = (4, 8, 16, 32, 64, 128)
    return [I(rng.choice(bits)), I(rng.choice(bits)), L(layout(rng))]


def gen_max_common(rng):
    a = layout(rng)
    r = rng.random()
    if r < 0.4:
        b = a
    elif r < 0.8:
        b = (a[0], compact(rng, a[0]) if rng.random() < 0.5 else loose(rng, a[0]))
    else:
        b = layout(rng)
    return [L(a), L(b)]


def gen_two_layouts(rng):
    return [L(layout(rng)), L(layout(rng))]


def gen_group_modes(rng):
    a = layout(rng, ranks=(2, 3, 3, 4), bare=0.0)
    r = len(a[0])
    if rng.random() < 0.9:
        b = rng.randrange(r)
        e = rng.randint(b + 1, r)
    else:
        b, e = rng.randrange(r + 2), rng.randrange(r + 3)
    return [L(a), I(b), I(e)]


def gen_select(rng):
    a = layout(rng, ranks=(2, 3, 3, 4), bare=0.0)
    r = len(a[0])
    modes = [rng.randrange(r if rng.random() < 0.9 else r + 2) for _ in range(rng.randint(1, r))]
    return [L(a), T(tuple(modes))]


def gen_slice(rng):
    a = layout(rng, ranks=(2, 3, 3, 4), bare=0.0)
    coord = tuple(None if isinstance(m, tuple) or rng.random() < 0.5 else rng.randrange(m) for m in a[0])
    return [L(a), T(coord)]


def gen_ordered(rng):
    shape = tuple(rng.choice(EXTENTS) for _ in range(rng.randint(1, 5)))
    order = list(range(len(shape)))
    rng.shuffle(order)
    return [T(shape), T(tuple(order))]


def gen_congruent(rng):
    a = rand_shape(rng)
    b = like(a, iter([rng.choice(EXTENTS) for _ in flat(a)])) if rng.random() < 0.5 else rand_shape(rng)
    return [T(a), T(b)]


def gen_swizzle(rng):
    bits = rng.choice((1, 2, 3))
    base = rng.randrange(5)
    shift = rng.choice((bits, bits + 1, bits + 2, -bits, -bits - 1))
    return [I(bits), I(base), I(shift), L(layout(rng, "bij", ranks=(2, 2, 3)))]


GENERATORS = [
    ("size", one_layout()),
    ("cosize", one_layout()),
    ("rank", one_layout()),
    ("depth", one_layout()),
    ("values", one_layout()),
    ("coalesce", one_layout()),
    ("flatten", one_layout()),
    ("filter", one_layout(("any", "any", "inj"))),
    ("filter_zeros", one_layout(("any", "any", "inj"))),
    ("nullspace", one_layout(("any", "any", "inj"))),
    ("right_inverse", one_layout(("bij", "bij", "inj", "any"))),
    ("left_inverse", one_layout(("bij", "bij", "inj", "any"))),
    ("idx2crd", gen_idx2crd),
    ("crd2idx", gen_crd2idx),
    ("composition", gen_composition),
    ("complement", gen_complement),
    ("logical_divide", gen_logical_divide),
    ("zipped_divide", gen_tuple_divide),
    ("tiled_divide", gen_tuple_divide),
    ("flat_divide", gen_tuple_divide),
    ("logical_product", gen_product),
    ("blocked_product", gen_product),
    ("raked_product", gen_product),
    ("zipped_product", gen_product),
    ("tiled_product", gen_product),
    ("flat_product", gen_product),
    ("tile_to_shape", gen_tile_to_shape),
    ("make_layout_tv", gen_layout_tv),
    ("recast_layout", gen_recast),
    ("max_common_vector", gen_max_common),
    ("max_common_layout", gen_max_common),
    ("append", gen_two_layouts),
    ("prepend", gen_two_layouts),
    ("group_modes", gen_group_modes),
    ("select", gen_select),
    ("dice", gen_slice),
    ("slice", gen_slice),
    ("make_ordered_layout", gen_ordered),
    ("is_congruent", gen_congruent),
    ("is_weakly_congruent", gen_congruent),
    ("swizzle_values", gen_swizzle),
]


# ------------------------------------------------------------ CuTe DSL

VALUES = 48  # indices evaluated by `values` and `swizzle_values`


def dsl(arg):
    kind, x = arg
    if kind == "L":
        return cute.make_layout(x[0], stride=x[1])
    if kind == "LL":
        return tuple(cute.make_layout(s, stride=d) for s, d in x)
    return x


def values(lay, n):
    return [lay(i) for i in range(min(n, VALUES))]


CALLS = {
    "size": cute.size,
    "cosize": cute.cosize,
    "rank": cute.rank,
    "depth": cute.depth,
    "coalesce": cute.coalesce,
    "flatten": cute.flatten,
    "filter": cute.filter,
    "filter_zeros": cute.filter_zeros,
    "nullspace": cute.nullspace,
    "right_inverse": cute.right_inverse,
    "left_inverse": cute.left_inverse,
    "idx2crd": cute.idx2crd,
    "crd2idx": cute.crd2idx,
    "composition": cute.composition,
    "complement": cute.complement,
    "logical_divide": cute.logical_divide,
    "zipped_divide": cute.zipped_divide,
    "tiled_divide": cute.tiled_divide,
    "flat_divide": cute.flat_divide,
    "logical_product": cute.logical_product,
    "blocked_product": cute.blocked_product,
    "raked_product": cute.raked_product,
    "zipped_product": cute.zipped_product,
    "tiled_product": cute.tiled_product,
    "flat_product": cute.flat_product,
    "tile_to_shape": cute.tile_to_shape,
    "make_layout_tv": cute.make_layout_tv,
    "recast_layout": cute.recast_layout,
    "max_common_vector": cute.max_common_vector,
    "max_common_layout": cute.max_common_layout,
    "append": cute.append,
    "prepend": cute.prepend,
    "group_modes": cute.group_modes,
    "select": lambda lay, modes: cute.select(lay, mode=list(modes)),
    "dice": cute.dice,
    "slice": cute.slice_,
    "make_ordered_layout": lambda shape, order: cute.make_ordered_layout(shape, order=order),
    "is_congruent": cute.is_congruent,
    "is_weakly_congruent": cute.is_weakly_congruent,
}


def call(op, args):
    if op == "values":
        return values(dsl(args[0]), size(args[0][1]))
    if op == "swizzle_values":
        bits, base, shift, lay = (dsl(a) for a in args)
        composed = cute.make_composed_layout(cute.make_swizzle(bits, base, shift), 0, lay)
        return values(composed, size(args[3][1]))
    return CALLS[op](*(dsl(a) for a in args))


CASES = []
START = 0  # the first case the traced function answers
PART = ""  # where it writes each answer as soon as it has it


class Recorded(Exception):
    """Raised once every case is answered, so that nothing is compiled."""


def answer_all():
    with open(PART, "a", buffering=1) as part:
        for n in range(START, len(CASES)):
            op, args = CASES[n]
            try:
                result = text(call(op, args))
            except Exception:
                result = "ERR"
            part.write(result + "\n")
            if n % 100 == 99:
                # A refused call must not disturb the ones after it.
                try:
                    ok = text(cute.coalesce(cute.make_layout((2, (3, 4)), stride=(1, (2, 6))))) == "24:1"
                except Exception:
                    ok = False
                if not ok:
                    raise SystemExit(f"a known call went wrong after case {n}")


@cute.jit
def traced():
    # The layout algebra only answers while a function is being traced.
    answer_all()
    raise Recorded()


def answered():
    with open(PART) as f:
        return sum(1 for _ in f)


def record():
    """Answer every case, in child processes: some calls crash CuTe DSL
    (a floating-point exception), and that is recorded as `CRASH`."""
    global START
    open(PART, "w").close()
    while (done := answered()) < len(CASES):
        START = done
        pid = os.fork()
        if pid == 0:
            try:
                traced()
            except BaseException:
                pass
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        now = answered()
        if now < len(CASES):
            if not os.WIFSIGNALED(status):
                raise SystemExit(f"the trace stopped at case {now} without crashing")
            with open(PART, "a") as part:
                part.write("CRASH\n")
    with open(PART) as f:
        results = [line.rstrip("\n") for line in f]
    os.remove(PART)
    return results


def arg_text(arg):
    kind, x = arg
    if kind == "L":
        return text(x[0]) + ":" + text(x[1])
    if kind == "LL":
        return ";".join(text(s) + ":" + text(d) for s, d in x)
    return text(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--per-op", type=int, default=400)
    opts = ap.parse_args()

    for k, (op, gen) in enumerate(GENERATORS):
        rng = random.Random(opts.seed * 1000 + k)
        for _ in range(opts.per_op):
            CASES.append((op, gen(rng)))

    global PART
    PART = opts.out + ".part"
    results = record()

    import cutlass

    with open(opts.out, "w") as f:
        f.write(f"# cute-dsl {getattr(cutlass, '__version__', '?')} seed {opts.seed} per-op {opts.per_op}\n")
        for (op, args), result in zip(CASES, results):
            f.write("\t".join([op] + [arg_text(a) for a in args] + ["=>", result]) + "\n")
    refused = sum(r == "ERR" for r in results)
    crashed = sum(r == "CRASH" for r in results)
    print(f"{len(CASES)} cases: CuTe DSL refused {refused} and crashed on {crashed} -> {opts.out}")


if __name__ == "__main__":
    main()
