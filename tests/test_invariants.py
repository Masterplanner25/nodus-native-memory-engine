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


class TestNoBytecodeVersionChange:
    def test_nodus_bytecode_version_still_4(self):
        try:
            import sys
            sys.path.insert(0, "C:/dev/Coding Language/src")
            from nodus.compiler.compiler import BYTECODE_VERSION
            assert BYTECODE_VERSION == 4
        except ImportError:
            pytest.skip("nodus-lang not on path")


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
