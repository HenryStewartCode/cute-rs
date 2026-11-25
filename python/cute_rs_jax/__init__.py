# SPDX-License-Identifier: MIT OR Apache-2.0
"""JAX shims for cute-rs kernels exported through the XLA FFI handler (``cute::xla_ffi``).

A cute-rs ``cdylib`` built with ``cute::xla_ffi_export!()`` exports the handler
``cute_xla_cutlass_call`` and a function that registers its kernel targets. This module loads
it, registers the handler with ``jax.ffi.register_ffi_target`` and gives ``cutlass_call``, a
thin wrapper over ``jax.ffi.ffi_call`` that names the target and passes the layouts.

Plain files, no build step: ``PYTHONPATH=python python your_script.py``. Sharding
(``shard_map``, ``custom_partitioning``) stays on the JAX side: wrap ``cutlass_call`` in them
as for any ``ffi_call``.
"""

from __future__ import annotations

import ctypes
from typing import Any, Sequence

TARGET_ATTR = "cute_target"
FFI_PREFIX = "cute_rs_cutlass_call"

# XLA dtype names <-> the cute-rs element types (``cute::xla_ffi::DTYPE_TABLE``).
_JAX_TO_CUTE = {
    "bool": "Boolean", "int4": "Int4", "int8": "Int8", "int16": "Int16", "int32": "Int32",
    "int64": "Int64", "uint8": "Uint8", "uint16": "Uint16", "uint32": "Uint32",
    "uint64": "Uint64", "float16": "Float16", "bfloat16": "BFloat16", "float32": "Float32",
    "float64": "Float64", "float8_e5m2": "Float8E5M2", "float8_e4m3fn": "Float8E4M3FN",
    "float8_e4m3b11fnuz": "Float8E4M3B11FNUZ", "float8_e4m3": "Float8E4M3",
    "float4_e2m1fn": "Float4E2M1FN", "float8_e8m0fnu": "Float8E8M0FNU",
}
_CUTE_TO_JAX = {v: k for k, v in _JAX_TO_CUTE.items()}

_libs: dict[str, ctypes.CDLL] = {}
_registered: set[str] = set()


def get_cutlass_call_ffi_version(jax_version: Sequence[int] | None = None) -> int:
    """2 for JAX >= 0.9.1, otherwise 1."""
    if jax_version is None:
        import jax

        jax_version = jax.version.__version_info__
    return 2 if tuple(jax_version)[:3] >= (0, 9, 1) else 1


def get_cutlass_call_ffi_name(allow_cuda_graph: bool = True, version: int | None = None) -> str:
    """The FFI target the handler is registered as (mirrors ``cute::xla_ffi``)."""
    if version is None:
        version = get_cutlass_call_ffi_version()
    return f"{FFI_PREFIX}{'' if allow_cuda_graph else '_nograph'}_v{version}"


def jax_to_cutlass_dtype(dtype: Any) -> str:
    """The cute-rs type name of a JAX/numpy dtype."""
    import numpy as np

    name = np.dtype(dtype).name if not isinstance(dtype, str) else dtype
    try:
        return _JAX_TO_CUTE[name]
    except KeyError:
        raise TypeError(f"{name} has no cute-rs element type") from None


def cutlass_to_jax_dtype(name: str) -> str:
    """The JAX dtype name of a cute-rs type name."""
    try:
        return _CUTE_TO_JAX[name]
    except KeyError:
        raise TypeError(f"{name} has no JAX dtype") from None


def jax_to_cutlass_layout_order(minor_to_major: Sequence[int]) -> tuple[int, ...]:
    """XLA's minor-to-major layout as cute's stride order (the inverse permutation)."""
    n = len(minor_to_major)
    if sorted(minor_to_major) != list(range(n)):
        raise ValueError(f"{tuple(minor_to_major)} is not a permutation of range({n})")
    out = [0] * n
    for rank, dim in enumerate(minor_to_major):
        out[dim] = rank
    return tuple(out)


cutlass_to_jax_layout_order = jax_to_cutlass_layout_order  # the inverse permutation of the inverse


def row_major(rank: int) -> tuple[int, ...]:
    return tuple(range(rank - 1, -1, -1))


def register_ffi(library: str, *, register_targets: str = "cute_rs_register_targets",
                 allow_cuda_graph: bool = True) -> ctypes.CDLL:
    """Load ``library``, register its handler with JAX and its kernel targets with cute-rs."""
    import jax

    if library in _libs:
        return _libs[library]
    lib = ctypes.CDLL(library)
    symbol = "cute_xla_cutlass_call" if allow_cuda_graph else "cute_xla_cutlass_call_nograph"
    handler = getattr(lib, symbol)
    name = get_cutlass_call_ffi_name(allow_cuda_graph)
    jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(handler), platform="CUDA")
    getattr(lib, register_targets)()
    _libs[library] = lib
    _registered.add(name)
    return lib


def is_ffi_registered(name: str | None = None) -> bool:
    """Whether this module registered the handler (``name``, or the default one)."""
    return (name or get_cutlass_call_ffi_name()) in _registered


def get_export_disabled_safety_checks():
    """The custom-call targets ``jax.export`` must be told to accept for exported functions
    that call a cute-rs kernel."""
    import jax

    return [jax.export.DisabledSafetyCheck.custom_call(n) for n in sorted(_registered)]


def cutlass_call(target: str, output_shape_dtype, *, input_layouts=None, output_layouts=None,
                 input_output_aliases=None, allow_cuda_graph: bool = True, **attrs):
    """A callable running the registered cute-rs target ``target`` on its arguments.

    ``output_shape_dtype`` is a ``jax.ShapeDtypeStruct`` or a sequence of them; layouts are
    XLA minor-to-major tuples (row-major by default); ``input_output_aliases`` is the
    ``{argument: result}`` map of ``ffi_call``, passed on to XLA and declared to the
    target as ``alias`` (a target that aliases checks XLA honoured it); scalar keyword
    attributes become scalar kernel arguments after the tensors.
    """
    import jax
    import numpy as np

    name = get_cutlass_call_ffi_name(allow_cuda_graph)
    if name not in _registered:
        raise RuntimeError("call cute_rs_jax.register_ffi(<library>) first")
    outs = output_shape_dtype if isinstance(output_shape_dtype, (list, tuple)) else [output_shape_dtype]
    call_attrs = {TARGET_ATTR: target, **attrs}
    if input_output_aliases:
        call_attrs["alias"] = ",".join(f"{a}:{r}" for a, r in sorted(input_output_aliases.items()))

    def run(*args):
        a = dict(call_attrs)
        in_layouts = input_layouts or [row_major(x.ndim) for x in args]
        out_layouts = output_layouts or [row_major(len(o.shape)) for o in outs]
        for i, m in enumerate(in_layouts):
            a[f"arg_m2m_{i}"] = np.asarray(m, dtype=np.int64)
        for i, m in enumerate(out_layouts):
            a[f"ret_m2m_{i}"] = np.asarray(m, dtype=np.int64)
        return jax.ffi.ffi_call(
            name,
            output_shape_dtype,
            input_layouts=[tuple(m) for m in in_layouts],
            output_layouts=[tuple(m) for m in out_layouts] if len(outs) > 1 or isinstance(output_shape_dtype, (list, tuple)) else tuple(out_layouts[0]),
            input_output_aliases=input_output_aliases,
        )(*args, **a)

    return run
