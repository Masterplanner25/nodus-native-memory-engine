"""Standing invariants for nodus-native-memory-engine v0.1."""
from __future__ import annotations

import math
import pytest


class TestNativeExtensionLoaded:
    def test_is_native_true(self):
        from nodus_native_memory_engine import is_native
        assert is_native(), "Rust extension not loaded — run maturin develop first"

    def test_core_module_importable(self):
        from nodus_native_memory_engine import _core
        assert _core is not None

    def test_core_version(self):
        from nodus_native_memory_engine import _core
        assert _core.__version__ == "0.1.0"


class TestAllFunctionsExported:
    EXPECTED = [
        "cosine_similarity",
        "batch_cosine_similarity",
        "compute_weight",
        "batch_compute_weights",
        "argsort_by_weight",
        "traverse_chain",
        "would_create_cycle",
        "rank_by_similarity",
        "rank_blended",
        "is_native",
    ]

    def test_all_functions_in_all(self):
        import nodus_native_memory_engine as e
        for fn_name in self.EXPECTED:
            assert fn_name in e.__all__, f"{fn_name} missing from __all__"

    def test_all_functions_callable(self):
        import nodus_native_memory_engine as e
        for fn_name in self.EXPECTED:
            assert callable(getattr(e, fn_name)), f"{fn_name} not callable"


class TestNativeAndPythonProduceSameResults:
    """Native Rust results must be within float precision of Python fallback."""

    def test_cosine_similarity_precision(self):
        from nodus_native_memory_engine import cosine_similarity, _py_cosine_similarity
        pairs = [
            ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]),
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.5, 0.5], [0.5, 0.5]),
        ]
        for a, b in pairs:
            n = cosine_similarity(a, b)
            p = _py_cosine_similarity(a, b)
            assert abs(n - p) < 1e-10, f"mismatch for {a}, {b}: {n} vs {p}"

    def test_compute_weight_precision(self):
        from nodus_native_memory_engine import compute_weight, _py_compute_weight
        cases = [(1.0, 0, 0, 0), (1.0, 10, 8, 2), (2.5, 100, 70, 30)]
        for c in cases:
            assert abs(compute_weight(*c) - _py_compute_weight(*c)) < 1e-10

    def test_traverse_chain_identical(self):
        from nodus_native_memory_engine import traverse_chain, _py_traverse_chain
        m = {"root": None, "mid": "root", "leaf": "mid"}
        assert traverse_chain(m, "leaf", 10) == _py_traverse_chain(m, "leaf", 10)

    def test_would_create_cycle_identical(self):
        from nodus_native_memory_engine import would_create_cycle, _py_would_create_cycle
        m = {"a": None, "b": "a", "c": "b"}
        for child, parent in [("d", "c"), ("c", "a"), ("b", "c")]:
            n = would_create_cycle(m, child, parent)
            p = _py_would_create_cycle(m, child, parent)
            assert n == p


class TestWeightFormulaCorrectness:
    """The weight formula must match the spec exactly."""

    def test_formula_impact_x_ratio_x_log(self):
        from nodus_native_memory_engine import compute_weight
        impact, usage, success, failure = 1.5, 20, 15, 5
        total = success + failure
        ratio = success / total
        expected = impact * (1.0 + ratio) * math.log1p(usage)
        assert abs(compute_weight(impact, usage, success, failure) - expected) < 1e-9

    def test_zero_usage_always_zero(self):
        from nodus_native_memory_engine import compute_weight
        assert compute_weight(100.0, 0, 50, 50) == 0.0
        assert compute_weight(0.001, 0, 0, 0) == 0.0


# Run in a fresh interpreter: import every module of `argv[1]` (none if empty),
# then print the opcodes a new nodus VM dispatches.
_OPCODES_AFTER_IMPORT = r"""
import importlib, json, pkgutil, sys
if sys.argv[1]:
    package = importlib.import_module(sys.argv[1])
    for info in pkgutil.walk_packages(package.__path__, sys.argv[1] + "."):
        try:
            importlib.import_module(info.name)
        except ImportError:
            pass  # an optional dependency this environment lacks
from nodus.vm.vm import VM
print(json.dumps(sorted(VM([], {}, code_locs=[], source_path=None)._dispatch)))
"""


def _opcodes_after_importing(package: str) -> list:
    import json
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    proc = subprocess.run([sys.executable, "-c", _OPCODES_AFTER_IMPORT, package],
                          capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr[-800:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestNoNewOpcodes:
    def test_importing_the_package_leaves_the_opcode_set_alone(self):
        """nodus-native-memory-engine adds no opcodes.

        This asserted nodus-lang's `BYTECODE_VERSION == 4`, reaching for a
        hard-coded checkout path to find it. That is nodus-lang's number, which
        this package cannot change, so it guarded nothing here and failed when
        nodus-lang 5.17.0 added opcodes through its own extension process
        (#969). It checks the package instead: importing every module of it
        leaves a nodus VM's opcode set as nodus-lang defines it, and the source
        never reaches for the VM's dispatch table or the bytecode version.
        """
        pytest.importorskip("nodus.vm.vm", reason="nodus-lang not importable")
        import re
        from pathlib import Path

        bare = _opcodes_after_importing("")
        assert len(bare) > 20, f"read {len(bare)} opcodes: the comparison would prove nothing"
        assert _opcodes_after_importing("nodus_native_memory_engine") == bare, (
            "importing nodus_native_memory_engine changed the VM's opcode set")
        vm_internals = re.compile(r"\b(?:vm|VM)\._dispatch\b|_build_dispatch_table|BYTECODE_VERSION\s*=")
        source = Path(__file__).resolve().parents[1] / "nodus_native_memory_engine"
        touching = [p.name for p in source.rglob("*.py")
                    if vm_internals.search(p.read_text(encoding="utf-8"))]
        assert not touching, f"{touching} reach into the VM's dispatch table"


class TestInstallRoundtrip:
    def test_package_importable(self):
        import nodus_native_memory_engine  # noqa: F401

    def test_version_is_semver(self):
        import re
        import nodus_native_memory_engine as e
        assert re.match(r"^\d+\.\d+\.\d+$", e.__version__)

    def test_version_matches_metadata(self):
        import importlib.metadata
        import nodus_native_memory_engine as e
        meta_version = importlib.metadata.version("nodus-native-memory-engine")
        assert e.__version__ == meta_version
