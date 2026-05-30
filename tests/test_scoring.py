"""Weight scoring — native and Python fallback must agree."""
from __future__ import annotations

import math
import pytest
from nodus_native_memory_engine import (
    compute_weight,
    batch_compute_weights,
    argsort_by_weight,
    _py_compute_weight,
    _py_batch_compute_weights,
    _py_argsort_by_weight,
)


class TestComputeWeight:
    def test_zero_usage_gives_zero_weight(self):
        assert compute_weight(1.0, 0, 0, 0) == 0.0

    def test_positive_usage_gives_positive_weight(self):
        assert compute_weight(1.0, 10, 5, 5) > 0.0

    def test_known_formula(self):
        impact, usage, success, failure = 1.0, 10, 8, 2
        total = success + failure
        ratio = success / total
        expected = impact * (1.0 + ratio) * math.log1p(usage)
        assert abs(compute_weight(impact, usage, success, failure) - expected) < 1e-9

    def test_higher_success_ratio_gives_higher_weight(self):
        good = compute_weight(1.0, 10, 9, 1)
        bad = compute_weight(1.0, 10, 1, 9)
        assert good > bad

    def test_higher_impact_gives_higher_weight(self):
        hi = compute_weight(2.0, 5, 0, 0)
        lo = compute_weight(0.5, 5, 0, 0)
        assert hi > lo

    def test_more_usage_gives_higher_weight(self):
        more = compute_weight(1.0, 100, 0, 0)
        less = compute_weight(1.0, 10, 0, 0)
        assert more > less

    def test_no_feedback_no_ratio_penalty(self):
        # success_ratio = 0 when no feedback; weight = impact * 1.0 * log1p(usage)
        w = compute_weight(1.0, 10, 0, 0)
        expected = math.log1p(10)
        assert abs(w - expected) < 1e-9


class TestBatchComputeWeights:
    def test_empty_batch(self):
        assert batch_compute_weights([]) == []

    def test_single_item(self):
        result = batch_compute_weights([(1.0, 10, 8, 2)])
        assert len(result) == 1
        assert abs(result[0] - compute_weight(1.0, 10, 8, 2)) < 1e-9

    def test_multiple_items_order_preserved(self):
        nodes = [(1.0, 5, 4, 1), (2.0, 10, 0, 0), (0.5, 100, 50, 50)]
        result = batch_compute_weights(nodes)
        assert len(result) == 3
        for i, (imp, u, s, f) in enumerate(nodes):
            assert abs(result[i] - compute_weight(imp, u, s, f)) < 1e-9


class TestArgsortByWeight:
    def test_empty(self):
        assert argsort_by_weight([]) == []

    def test_single(self):
        assert argsort_by_weight([(1.0, 10, 5, 5)]) == [0]

    def test_order_high_to_low(self):
        nodes = [
            (1.0, 1, 0, 0),   # lowest weight
            (1.0, 100, 90, 10),  # highest weight
            (1.0, 10, 5, 5),   # middle
        ]
        indices = argsort_by_weight(nodes)
        weights = batch_compute_weights(nodes)
        assert indices[0] == 1  # highest first
        assert indices[-1] == 0  # lowest last

    def test_all_zero_usage_stable(self):
        nodes = [(1.0, 0, 0, 0), (1.0, 0, 0, 0)]
        indices = argsort_by_weight(nodes)
        assert set(indices) == {0, 1}


class TestFallbackMatchesNative:
    def test_compute_weight_matches(self):
        cases = [
            (1.0, 0, 0, 0),
            (1.0, 10, 8, 2),
            (2.5, 100, 90, 10),
            (0.1, 1, 0, 1),
        ]
        for c in cases:
            assert abs(compute_weight(*c) - _py_compute_weight(*c)) < 1e-10

    def test_batch_matches(self):
        nodes = [(1.0, 5, 3, 2), (2.0, 0, 0, 0), (0.5, 50, 40, 10)]
        native = batch_compute_weights(nodes)
        python = _py_batch_compute_weights(nodes)
        for n, p in zip(native, python):
            assert abs(n - p) < 1e-10

    def test_argsort_matches(self):
        nodes = [(1.0, 5, 3, 2), (2.0, 100, 80, 20), (0.5, 1, 0, 0)]
        native = argsort_by_weight(nodes)
        python = _py_argsort_by_weight(nodes)
        assert native == python
