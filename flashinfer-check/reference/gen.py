# SPDX-License-Identifier: Apache-2.0
"""Run FlashInfer on fixed inputs and save the inputs and its outputs as case files.

    python gen.py <cases-dir> [module ...]

Each `cases_<module>.py` beside this file has a `cases()` generator of `common.Case`s. With
no module named, every one runs. The inputs are seeded, so a rerun writes the same files.
"""
import glob
import importlib
import os
import sys
import traceback

import torch

import flashinfer


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    # FlashInfer compiles its kernels with ninja, which pip installs beside the interpreter.
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    import common
    root = sys.argv[1]
    names = sys.argv[2:] or sorted(
        os.path.basename(p)[len("cases_") : -len(".py")] for p in glob.glob(os.path.join(here, "cases_*.py"))
    )
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "reference.txt"), "w") as f:
        f.write(f"flashinfer {flashinfer.__version__}\ntorch {torch.__version__}\n{torch.cuda.get_device_name()}\n")
    failed = 0
    for name in names:
        module = importlib.import_module(f"cases_{name}")
        torch.manual_seed(0)
        count = 0
        common.REFUSED.clear()
        try:
            for case in module.cases():
                assert case.meta["module"] == name, (case.name, name)
                case.write(root)
                count += 1
        except Exception:
            failed += 1
            traceback.print_exc()
        listing = os.path.join(root, f"refused_{name}.txt")
        if common.REFUSED:
            with open(listing, "w") as f:
                f.writelines(f"{case}\t{why}\n" for case, why in common.REFUSED)
        elif os.path.exists(listing):
            os.remove(listing)
        print(f"{name}: {count} cases, {len(common.REFUSED)} refused by FlashInfer", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
