"""Combined ranking pipelines — rank_by_similarity and rank_blended."""
from __future__ import annotations

import pytest
from nodus_native_memory_engine import (
    rank_by_similarity,
    rank_blended,
    _py_rank_by_similarity,
    _py_rank_blended,
)


class TestRankBySimilarity:
    def test_empty_embeddings(self):
        assert rank_by_similarity([1.0, 0.0], [], threshold=0.0, top_k=5) == []

    def test_single_above_threshold(self):
        result = rank_by_similarity(
            [1.0, 0.0],
            [("node-a", [1.0, 0.0])],
            threshold=0.9,
            top_k=5,
        )
        assert result == ["node-a"]

    def test_below_threshold_excluded(self):
        result = rank_by_similarity(
            [1.0, 0.0],
            [("node-b", [0.0, 1.0])],   # orthogonal → similarity = 0
            threshold=0.5,
            top_k=5,
        )
        assert result == []

    def test_ordering_best_first(self):
        q = [1.0, 0.0]
        embeddings = [
            ("low", [0.6, 0.8]),   # sim ≈ 0.6
            ("high", [1.0, 0.0]),  # sim = 1.0
            ("mid", [0.8, 0.6]),   # sim ≈ 0.8
        ]
        result = rank_by_similarity(q, embeddings, threshold=0.0, top_k=10)
        assert result.index("high") < result.index("mid") < result.index("low")

    def test_top_k_limits_results(self):
        q = [1.0, 0.0]
        embeddings = [(f"n{i}", [1.0, 0.0]) for i in range(20)]
        result = rank_by_similarity(q, embeddings, threshold=0.0, top_k=5)
        assert len(result) == 5

    def test_zero_query_returns_empty_via_fallback_logic(self):
        # All-zeros query — similarity is 0 for all, so all below any positive threshold
        result = rank_by_similarity(
            [0.0, 0.0],
            [("n", [1.0, 0.0])],
            threshold=0.01,
            top_k=5,
        )
        assert result == []


class TestRankBlended:
    def test_empty_nodes(self):
        assert rank_blended([], [1.0, 0.0], 0.5, 5) == []

    def test_weight_only_sim_weight_zero(self):
        nodes = [
            ("low", 1.0, 1, 0, 0, None),    # low weight
            ("high", 1.0, 100, 90, 10, None),  # high weight
        ]
        result = rank_blended(nodes, [], 0.0, 10)
        assert result.index("high") < result.index("low")

    def test_sim_only_sim_weight_one(self):
        q = [1.0, 0.0]
        nodes = [
            ("orthogonal", 5.0, 1000, 900, 100, [0.0, 1.0]),  # sim=0, high weight
            ("aligned", 0.1, 0, 0, 0, [1.0, 0.0]),            # sim=1, low weight
        ]
        result = rank_blended(nodes, q, 1.0, 10)
        assert result.index("aligned") < result.index("orthogonal")

    def test_blended_50_50(self):
        q = [1.0, 0.0]
        nodes = [
            ("a", 1.0, 10, 5, 5, [1.0, 0.0]),    # good sim, decent weight
            ("b", 10.0, 100, 90, 10, [0.0, 1.0]), # poor sim, excellent weight
        ]
        # At 50/50 blend, b's high weight should overcome its poor sim
        result = rank_blended(nodes, q, 0.5, 10)
        # Both should appear; b likely first due to weight dominance
        assert len(result) == 2

    def test_top_k_respected(self):
        nodes = [("n" + str(i), 1.0, 1, 0, 0, None) for i in range(20)]
        result = rank_blended(nodes, [], 0.0, 5)
        assert len(result) == 5

    def test_none_embedding_handled(self):
        nodes = [("no-emb", 1.0, 10, 8, 2, None)]
        result = rank_blended(nodes, [1.0, 0.0], 0.5, 5)
        assert "no-emb" in result


class TestFallbackMatchesNative:
    def test_rank_by_similarity_matches(self):
        q = [0.7, 0.3, 0.1]
        embeddings = [
            ("a", [0.9, 0.1, 0.0]),
            ("b", [0.1, 0.9, 0.0]),
            ("c", [0.5, 0.5, 0.5]),
        ]
        native = rank_by_similarity(q, embeddings, threshold=0.0, top_k=10)
        python = _py_rank_by_similarity(q, embeddings, threshold=0.0, top_k=10)
        assert native == python

    def test_rank_blended_matches(self):
        q = [1.0, 0.0]
        nodes = [
            ("a", 1.0, 10, 5, 5, [1.0, 0.0]),
            ("b", 2.0, 5, 0, 0, [0.5, 0.5]),
            ("c", 0.5, 100, 80, 20, None),
        ]
        native = rank_blended(nodes, q, 0.5, 10)
        python = _py_rank_blended(nodes, q, 0.5, 10)
        assert native == python
