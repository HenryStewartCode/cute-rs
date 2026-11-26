# SPDX-License-Identifier: Apache-2.0
"""Compare what the cute-rs ports wrote with what FlashInfer wrote, case by case.

    python compare.py <cases-dir> [report.md] [--only <text>]

Prints one line per output and a summary; exits 1 if any output is missing or outside its
rule. `--only` keeps the cases whose name contains the text. Needs only PyTorch on the CPU.
"""
import os
import sys

import torch

from common import from_bytes, parse_shape, read_meta

# Sign-magnitude float formats: the bits below the sign order the magnitudes.
_BITS = {
    "float16": (torch.int16, 16),
    "bfloat16": (torch.int16, 16),
    "float32": (torch.int32, 32),
    "float64": (torch.int64, 64),
    "float8_e4m3fn": (torch.uint8, 8),
    "float8_e5m2": (torch.uint8, 8),
}


def ordinals(t, name):
    """Each value's position among its type's representable values (-0 and +0 share one)."""
    raw, bits = _BITS[name]
    b = t.contiguous().view(raw).to(torch.int64) & ((1 << bits) - 1) if bits < 64 else t.contiguous().view(raw)
    sign = b >> (bits - 1) & 1
    mag = b & ((1 << (bits - 1)) - 1)
    return torch.where(sign == 1, -mag, mag)


def compare(ref, out, name, rule):
    """(ok, mismatched elements, a description of the largest difference)."""
    total = ref.numel()
    if name not in _BITS:
        diff = (ref.to(torch.int64) - out.to(torch.int64)).abs()
        kind, *params = rule.split(":")
        if kind not in ("exact", "int"):
            raise ValueError(f"rule {rule} on integer type {name}")
        bad = int((diff > (int(params[0]) if kind == "int" else 0)).sum())
        differing = int((diff > 0).sum())
        worst = f"largest difference {int(diff.max()) if total else 0}, {differing} of {total} differ"
        return bad == 0, bad, worst
    rf, of = ref.to(torch.float64), out.to(torch.float64)
    both_nan = rf.isnan() & of.isnan()
    one_nan = rf.isnan() ^ of.isnan()
    ulps = (ordinals(ref, name) - ordinals(out, name)).abs()
    ulps = torch.where(both_nan, torch.zeros_like(ulps), ulps)
    err = torch.where(both_nan | one_nan, torch.zeros_like(rf), (rf - of).abs())
    err = torch.where(err.isnan(), torch.full_like(err, float("inf")), err)  # inf against inf of the other sign
    same_inf = rf.isinf() & (rf == of)
    err = torch.where(same_inf, torch.zeros_like(err), err)
    max_ulp = int(ulps.max()) if total else 0
    max_err = float(err.max()) if total else 0.0
    worst = f"max {max_ulp} ulp, {max_err:.3g} abs"
    kind, *params = rule.split(":")
    if kind == "exact":
        raw = _BITS[name][0]
        wrong = ref.contiguous().view(raw) != out.contiguous().view(raw)
    elif kind == "ulp":
        wrong = (ulps > int(params[0])) | one_nan
    else:
        atol, rtol = float(params[0]), float(params[1])
        wrong = (err > atol + rtol * rf.abs()) | one_nan
        wrong &= ~same_inf
    bad = int(wrong.sum())
    differing = int(((ulps > 0) | one_nan).sum())
    lone_nans = f", {int(one_nan.sum())} are NaN on one side only" if one_nan.any() else ""
    return bad == 0, bad, f"{worst}, {differing} of {total} differ{lone_nans}"


def main():
    args = sys.argv[1:]
    only = ""
    if "--only" in args:
        at = args.index("--only")
        only = args[at + 1]
        del args[at : at + 2]
    root = args[0]
    rows = []
    failed = missing = 0
    for case in sorted(os.listdir(root)):
        d = os.path.join(root, case)
        if only not in case or not os.path.isfile(os.path.join(d, "meta.txt")):
            continue
        meta = read_meta(d)
        error = None
        if os.path.exists(os.path.join(d, "error.txt")):
            with open(os.path.join(d, "error.txt")) as f:
                error = f.read().strip().splitlines()[0][:160]
        for key, value in meta.items():
            if not key.startswith("ref."):
                continue
            name = key[4:]
            dtype, shape, rule = value.split(" ")
            path = os.path.join(d, f"out.{name}.bin")
            if error or not os.path.exists(path):
                missing += 1
                rows.append((meta["module"], case, name, rule, "NOT RUN", error or "no output written"))
                continue
            with open(os.path.join(d, f"ref.{name}.bin"), "rb") as f:
                ref = f.read()
            with open(path, "rb") as f:
                out = f.read()
            if len(ref) != len(out):
                failed += 1
                rows.append((meta["module"], case, name, rule, "FAIL", f"{len(out)} bytes for {len(ref)}"))
                continue
            shape = parse_shape(shape)
            ref_t = from_bytes(ref, dtype, shape)
            ok, bad, worst = compare(ref_t, from_bytes(out, dtype, shape), dtype, rule)
            if ref_t.numel() > 1 and len(set(ref)) == 1:
                worst += "; FlashInfer's output is all one byte value, so this shows little"
            failed += not ok
            rows.append((meta["module"], case, name, rule, "ok" if ok else "FAIL", f"{worst}" + (f"; {bad} outside the rule" if bad else "")))
    width = max((len(r[1]) + len(r[2]) for r in rows), default=10) + 1
    for module, case, name, rule, status, detail in rows:
        print(f"{status:7} {case + '.' + name:{width}} {rule:16} {detail}")
    modules = sorted({r[0] for r in rows})
    print()
    print(f"{'module':14} {'outputs':>7} {'ok':>5} {'fail':>5} {'not run':>7}")
    summary = []
    for m in modules:
        mine = [r for r in rows if r[0] == m]
        counts = [len(mine)] + [sum(r[4] == s for r in mine) for s in ("ok", "FAIL", "NOT RUN")]
        summary.append((m, *counts))
        print(f"{m:14} {counts[0]:7} {counts[1]:5} {counts[2]:5} {counts[3]:7}")
    print(f"{'total':14} {len(rows):7} {len(rows) - failed - missing:5} {failed:5} {missing:7}")
    refused = []
    for listing in sorted(f for f in os.listdir(root) if f.startswith("refused_") and f.endswith(".txt")):
        with open(os.path.join(root, listing)) as f:
            refused += [line.rstrip("\n").split("\t", 1) for line in f if only in line.split("\t", 1)[0]]
    if refused:
        print(f"\nFlashInfer refused {len(refused)} calls, so they are not compared:")
        for case, why in refused:
            print(f"  {case}: {why}")
    if len(args) > 1:
        with open(args[1], "w") as f:
            f.write("| Module | Outputs | Match | Differ | Not run |\n| --- | ---: | ---: | ---: | ---: |\n")
            for m, n, ok, bad, skipped in summary:
                f.write(f"| `{m}` | {n} | {ok} | {bad} | {skipped} |\n")
            f.write("\n| | Case and output | Rule | Largest difference |\n| --- | --- | --- | --- |\n")
            for module, case, name, rule, status, detail in rows:
                f.write(f"| {status} | `{case}.{name}` | {rule} | {detail} |\n")
            if refused:
                f.write("\n| Call FlashInfer refused | Its error |\n| --- | --- |\n")
                for case, why in refused:
                    f.write(f"| `{case}` | {why} |\n")
    sys.exit(1 if failed or missing else 0)


if __name__ == "__main__":
    main()
