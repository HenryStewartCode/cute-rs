# SPDX-License-Identifier: MIT OR Apache-2.0
"""A cute-rs kernel inside jax.jit (the counterpart of the CuTe DSL tutorial's
``cutlass_call_basic.py``, which this does not copy), through the XLA FFI handler of
examples/xla_cutlass_call.rs.

    PYTHONPATH=python python examples/xla/cutlass_call_basic.py target/release/examples/libxla_cutlass_call.so
"""

import sys

import jax
import jax.numpy as jnp

import cute_rs_jax as cj


def main(path):
    cj.register_ffi(path)
    add_one = cj.cutlass_call("add_one", jax.ShapeDtypeStruct((1000,), jnp.float32))
    x = jnp.arange(1000, dtype=jnp.float32)
    y = jax.jit(add_one)(x)
    assert bool(jnp.all(y == x + 1)), y
    # An export carries the custom call; the target needs whitelisting.
    exported = jax.export.export(jax.jit(add_one), disabled_checks=cj.get_export_disabled_safety_checks())(x)
    assert "cute_rs_cutlass_call" in exported.mlir_module()
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1])
