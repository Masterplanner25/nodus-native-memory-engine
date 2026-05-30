"""Cosine similarity — native and Python fallback must agree."""
from __future__ import annotations

import math
import pytest
import nodus_native_memory_engine as e
from nodus_native_memory_engine import (
    cosine_similarity,
    batch_cosine_similarity,
    _py_cosine_similarity,
    _py_batch_cosine_similarity,
)


class TestCosineSimilarityNative:
    def test_identical_vectors(self):
        assert abs(cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9

    def test_opposite_vectors(self):
        assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-9

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_all_zeros_returns_zero(self):
        assert cosine_similarity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == 0.0

    def test_known_value(self):
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert abs(cosine_similarity(a, b) - expected) < 1e-9

    def test_length_mismatch_raises(self):
        with pytest.raises((ValueError, Exception)):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_symmetry(self):
        a = [0.3, 0.7, 0.1]
        b = [0.5, 0.5, 0.9]
        assert abs(cosine_similarity(a, b) - cosine_similarity(b, a)) < 1e-12

    def test_range_minus_one_to_one(self):
        for _ in range(20):
            import random
            a = [random.uniform(-1, 1) for _ in range(8)]
            b = [random.uniform(-1, 1) for _ in range(8)]
            sim = cosine_similarity(a, b)
            assert -1.0 - 1e-9 <= sim <= 1.0 + 1e-9


class TestBatchCosineSimilarity:
    def test_single_row(self):
        q = [1.0, 0.0]
        m = [[1.0, 0.0]]
        result = batch_cosine_similarity(q, m)
        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-9

    def test_multiple_rows(self):
        q = [1.0, 0.0]
        m = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        result = batch_cosine_similarity(q, m)
        assert abs(result[0] - 1.0) < 1e-9
        assert abs(result[1]) < 1e-9
        assert abs(result[2] - (-1.0)) < 1e-9

    def test_empty_matrix(self):
        assert batch_cosine_similarity([1.0, 0.0], []) == []

    def test_length_mismatch_raises(self):
        with pytest.raises(Exception):
            batch_cosine_similarity([1.0, 0.0], [[1.0, 0.0, 0.0]])


class TestFallbackMatchesNative:
    """Python fallback must produce results within float precision of native."""

    def test_cosine_similarity_matches(self):
        a = [0.3, 0.7, 0.1, 0.9]
        b = [0.5, 0.2, 0.8, 0.4]
        native = cosine_similarity(a, b)
        python = _py_cosine_similarity(a, b)
        assert abs(native - python) < 1e-10

    def test_batch_matches(self):
        q = [0.3, 0.7, 0.1]
        matrix = [[0.5, 0.2, 0.8], [0.9, 0.1, 0.3], [0.0, 0.0, 0.0]]
        native = batch_cosine_similarity(q, matrix)
        python = _py_batch_cosine_similarity(q, matrix)
        for n, p in zip(native, python):
            assert abs(n - p) < 1e-10
