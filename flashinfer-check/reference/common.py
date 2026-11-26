# SPDX-License-Identifier: Apache-2.0
"""The case files `gen.py` writes, `fi-check` reads and `compare.py` checks.

A case is a directory:

    meta.txt        `key=value` lines: `module`, the case's parameters, and one line per tensor
                    (`in.<name>=<dtype> <d0>x<d1>...`, `ref.<name>=<dtype> <shape> <rule>`)
    in.<name>.bin   an input, raw little-endian, C order
    ref.<name>.bin  what FlashInfer made of the inputs
    out.<name>.bin  what the cute-rs port made of them (written by `fi-check`)

`refused_<module>.txt` beside the case directories lists the calls FlashInfer would not run.

Rules say how close `out` must be to `ref`:

    exact           the same bytes
    ulp:N           at most N representable values of the tensor's type apart
    int:N           integers at most N apart (quantised values that round at a boundary)
    close:A:R       |out - ref| <= A + R * |ref|
"""
import os

import torch

# Types NumPy has no name for are saved through an integer view of the same width.
_VIEWS = {
    torch.bfloat16: torch.int16,
    torch.float8_e4m3fn: torch.uint8,
    torch.float8_e5m2: torch.uint8,
    torch.bool: torch.uint8,
}


def dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def to_bytes(t):
    t = t.detach().contiguous().cpu()
    if t.dtype in _VIEWS:
        t = t.view(_VIEWS[t.dtype])
    return t.numpy().tobytes()


def from_bytes(data, name, shape):
    """The tensor `to_bytes` saved, of type `name` and `shape`."""
    dtype = getattr(torch, name)
    raw = _VIEWS.get(dtype, dtype)
    t = torch.frombuffer(bytearray(data), dtype=raw)
    if raw is not dtype:
        t = t.view(dtype)
    return t.reshape(shape)


# (case name, why) of each call FlashInfer itself would not run; `gen.py` saves the list.
REFUSED = []


def refused(name, error):
    """Record that FlashInfer raised `error` on case `name` (a type or shape it does not take),
    so the case is listed as not compared and the rest of the module still runs."""
    why = f"{type(error).__name__}: {(str(error).splitlines() or [''])[0][:200]}"
    REFUSED.append((name, why))
    print(f"REFUSED {name}: {why}", flush=True)


class Case:
    """One call of a FlashInfer function: its parameters, inputs and FlashInfer's outputs."""

    def __init__(self, name, module, **meta):
        self.name = name
        self.meta = {"module": module, **meta}
        self.inputs = {}
        self.outputs = {}

    def input(self, name, tensor):
        """Save `tensor` as an input; returns it, so a call can be written around it."""
        assert name not in self.inputs, name
        self.inputs[name] = tensor.detach().clone()
        return tensor

    def output(self, name, tensor, rule):
        """Save `tensor`, a result of FlashInfer's, to be matched under `rule`."""
        assert name not in self.outputs, name
        kind = rule.split(":")[0]
        assert kind in ("exact", "ulp", "int", "close"), rule
        self.outputs[name] = (tensor.detach().clone(), rule)

    def write(self, root):
        d = os.path.join(root, self.name)
        os.makedirs(d, exist_ok=True)
        lines = []
        for k, v in self.meta.items():
            if isinstance(v, bool):
                v = int(v)
            elif isinstance(v, (list, tuple)):
                v = ",".join(str(x) for x in v)
            assert "\n" not in str(v) and "=" not in k
            lines.append(f"{k}={v}")
        for name, t in self.inputs.items():
            lines.append(f"in.{name}={dtype_name(t.dtype)} {_shape(t)}")
            with open(os.path.join(d, f"in.{name}.bin"), "wb") as f:
                f.write(to_bytes(t))
        for name, (t, rule) in self.outputs.items():
            lines.append(f"ref.{name}={dtype_name(t.dtype)} {_shape(t)} {rule}")
            with open(os.path.join(d, f"ref.{name}.bin"), "wb") as f:
                f.write(to_bytes(t))
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")


def _shape(t):
    return "x".join(str(n) for n in t.shape) if t.dim() else "scalar"


def read_meta(case_dir):
    meta = {}
    with open(os.path.join(case_dir, "meta.txt")) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                k, v = line.split("=", 1)
                meta[k] = v
    return meta


def parse_shape(s):
    return [] if s == "scalar" else [int(n) for n in s.split("x")]
