# SPDX-License-Identifier: (MIT OR Apache-2.0) AND BSD-3-Clause
#
# This script follows a CuTe DSL example of NVIDIA CUTLASS 4.8.1:
#   examples/python/CuTeDSL/dsl_tutorials/tvm_ffi/jit_and_use_in_torch.py,
#     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# and takes its steps (b = a + 1 on torch tensors of 10 Float32 elements, then again on
# a + 1) from it. Those parts are under the BSD-3-Clause licence, and the rest of the file
# is cute-rs's own, under MIT OR Apache-2.0. THIRD-PARTY-NOTICES.md has the licence text.
"""Call the cute-rs TVM-FFI function `add_one` (examples/tvm_ffi_add.rs) on torch tensors.

    python examples/tvm_ffi/jit_and_use_in_torch.py target/release/examples/libtvm_ffi_add.so
"""

import sys

import torch
import tvm_ffi


def main(path):
    mod = tvm_ffi.load_module(path)
    a = torch.arange(10, dtype=torch.float32, device="cuda")
    b = torch.zeros(10, dtype=torch.float32, device="cuda")
    mod.add_one(a, b)
    torch.cuda.synchronize()
    print("b after add_one(a, b):", b)
    assert torch.equal(b, a + 1), b

    # Any length: the extent is a runtime value.
    a = a + 1
    mod.add_one(a, b)
    torch.cuda.synchronize()
    print("b after add_one(a + 1, b):", b)
    assert torch.equal(b, a + 1), b
    c = torch.arange(37, dtype=torch.float32, device="cuda")
    d = torch.empty_like(c)
    mod.add_one(c, d)
    torch.cuda.synchronize()
    assert torch.equal(d, c + 1), d
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1])
