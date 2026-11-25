# SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
#
# This script follows a CuTe DSL example of NVIDIA CUTLASS 4.8.1:
#   examples/python/CuTeDSL/dsl_tutorials/tvm_ffi/error_reporting.py,
#     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# and takes its case (a function compiled for tensors of 10 elements called with one of 5)
# from it. Those parts are under the BSD-3-Clause licence, and the rest of the file is
# cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.
"""Errors from a cute-rs TVM-FFI function (examples/tvm_ffi_add.rs) as Python exceptions.

    python examples/tvm_ffi/error_reporting.py target/release/examples/libtvm_ffi_add.so
"""

import sys

import torch
import tvm_ffi


def expect(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type as e:
        print(f"{exc_type.__name__}: {e}")
        return str(e)
    raise AssertionError(f"expected {exc_type.__name__}")


def main(path):
    mod = tvm_ffi.load_module(path)
    a = torch.arange(10, dtype=torch.float32, device="cuda")
    b = torch.zeros(10, dtype=torch.float32, device="cuda")
    mod.add_one_static(a, b)
    torch.cuda.synchronize()
    assert torch.equal(b, a + 1), b

    # The static layout is (10,): a tensor of 5 elements is refused before the launch.
    m = expect(ValueError, mod.add_one_static, torch.arange(5, dtype=torch.float32, device="cuda"), b)
    assert "argument 0 (`a`)" in m, m
    # The element type, the device and the argument kinds are checked too.
    expect(ValueError, mod.add_one, a.half(), b)
    expect(ValueError, mod.add_one, a.cpu(), b)
    expect(TypeError, mod.add_one, a, 3)
    expect(TypeError, mod.add_one, a)
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1])
