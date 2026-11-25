# SPDX-License-Identifier: MIT OR Apache-2.0
"""Pure-Python checks of the JAX shim (no GPU); the XLA test needs jax and CUTE_RS_XLA_LIB."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import cute_rs_jax as cj  # noqa: E402


class Converters(unittest.TestCase):
    def test_layout_orders(self):
        self.assertEqual(cj.jax_to_cutlass_layout_order((1, 2, 0)), (2, 0, 1))
        self.assertEqual(cj.cutlass_to_jax_layout_order((2, 0, 1)), (1, 2, 0))
        with self.assertRaises(ValueError):
            cj.jax_to_cutlass_layout_order((0, 0))

    def test_names_and_dtypes(self):
        self.assertEqual(cj.get_cutlass_call_ffi_version((0, 9, 1)), 2)
        self.assertEqual(cj.get_cutlass_call_ffi_version((0, 9, 0)), 1)
        self.assertEqual(cj.get_cutlass_call_ffi_name(False, 1), "cute_rs_cutlass_call_nograph_v1")
        self.assertEqual(cj.jax_to_cutlass_dtype("bfloat16"), "BFloat16")
        self.assertEqual(cj.cutlass_to_jax_dtype("Float32"), "float32")


@unittest.skipUnless(os.environ.get("CUTE_RS_XLA_LIB"), "set CUTE_RS_XLA_LIB (and have jax with CUDA)")
class WithJax(unittest.TestCase):
    def test_basic(self):
        import runpy
        runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "..", "examples", "xla", "cutlass_call_basic.py"),
                       run_name="not_main")["main"](os.environ["CUTE_RS_XLA_LIB"])


if __name__ == "__main__":
    unittest.main()
