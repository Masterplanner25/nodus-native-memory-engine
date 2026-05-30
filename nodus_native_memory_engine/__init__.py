"""nodus-native-memory-engine — Rust-accelerated memory operations for Nodus.

All functions have pure-Python fallbacks so the library works even if the
Rust extension has not been compiled. Call ``is_native()`` to check.

Public API
----------
cosine_similarity(a, b)                        single-pair cosine similarity
batch_cosine_similarity(query, matrix)         one query vs many embeddings
compute_weight(impact, usage, success, fail)   single node weight
batch_compute_weights(nodes)                   batch weights
argsort_by_weight(nodes)                       indices sorted by weight desc
traverse_chain(id_to_parent, start, depth)     causal chain walk + cycle check
would_create_cycle(id_to_parent, child, par)   cycle pre-check
rank_by_similarity(query, embeddings, ...)     top-k by cosine similarity
rank_blended(nodes, query, sim_weight, top_k)  combined weight+similarity ranking
is_native()                                    True if Rust extension is loaded
"""
from __future__ import annotations

import math
from typing import Optional

__version__ = "0.1.0"

# ─── try to load the Rust extension ──────────────────────────────────────────

try:
    from nodus_native_memory_engine import _core as _native
    _NATIVE = True
except ImportError:
    _native = None
    _NATIVE = False


def is_native() -> bool:
    """Return True if the compiled Rust extension is available."""
    return _NATIVE


# ─── pure-Python fallbacks ────────────────────────────────────────────────────

def _py_cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vectors must have equal length: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _py_batch_cosine_similarity(query: list[float], matrix: list[list[float]]) -> list[float]:
    return [_py_cosine_similarity(query, row) for row in matrix]


def _py_compute_weight(
    impact_score: float, usage_count: int, success_count: int, failure_count: int
) -> float:
    total = success_count + failure_count
    success_ratio = success_count / total if total > 0 else 0.0
    return impact_score * (1.0 + success_ratio) * math.log1p(usage_count)


def _py_batch_compute_weights(nodes: list[tuple]) -> list[float]:
    return [_py_compute_weight(*n) for n in nodes]


def _py_argsort_by_weight(nodes: list[tuple]) -> list[int]:
    weights = [(_py_compute_weight(*n), i) for i, n in enumerate(nodes)]
    weights.sort(key=lambda x: x[0], reverse=True)
    return [i for _, i in weights]


def _py_traverse_chain(
    id_to_parent: dict[str, Optional[str]],
    start_id: str,
    max_depth: int,
) -> list[str]:
    chain = [start_id]
    seen = {start_id}
    current = start_id
    for _ in range(max_depth - 1):
        parent = id_to_parent.get(current)
        if parent is None:
            break
        if parent in seen:
            raise ValueError(f"causal cycle detected at node: {parent}")
        seen.add(parent)
        chain.append(parent)
        current = parent
    chain.reverse()
    return chain


def _py_would_create_cycle(
    id_to_parent: dict[str, Optional[str]],
    child_id: str,
    parent_id: str,
) -> bool:
    seen = {child_id}
    current = parent_id
    for _ in range(1000):
        if current == child_id:
            return True
        if current in seen:
            return False
        seen.add(current)
        next_node = id_to_parent.get(current)
        if next_node is None:
            return False
        current = next_node
    return False


def _py_rank_by_similarity(
    query: list[float],
    embeddings: list[tuple[str, list[float]]],
    threshold: float,
    top_k: int,
) -> list[str]:
    scored = []
    for node_id, emb in embeddings:
        sim = _py_cosine_similarity(query, emb)
        if sim >= threshold:
            scored.append((sim, node_id))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [node_id for _, node_id in scored[:top_k]]


def _py_rank_blended(
    nodes: list[tuple],
    query: list[float],
    sim_weight: float,
    top_k: int,
) -> list[str]:
    use_sim = bool(query) and sim_weight > 0.0
    scored = []
    for item in nodes:
        node_id, impact, usage, success, failure, emb = item
        w = _py_compute_weight(impact, usage, success, failure)
        sim = 0.0
        if use_sim and emb is not None and len(emb) == len(query):
            sim = _py_cosine_similarity(query, emb)
        blended = sim_weight * sim + (1.0 - sim_weight) * w
        scored.append((blended, node_id))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [node_id for _, node_id in scored[:top_k]]


# ─── public API (routes to Rust or Python fallback) ──────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. Returns 0.0 for zero vectors."""
    if _NATIVE:
        return _native.cosine_similarity(a, b)
    return _py_cosine_similarity(a, b)


def batch_cosine_similarity(query: list[float], matrix: list[list[float]]) -> list[float]:
    """Cosine similarity between *query* and each row in *matrix*."""
    if _NATIVE:
        return _native.batch_cosine_similarity(query, matrix)
    return _py_batch_cosine_similarity(query, matrix)


def compute_weight(
    impact_score: float,
    usage_count: int,
    success_count: int,
    failure_count: int,
) -> float:
    """Compute ranking weight: impact * (1 + success_ratio) * log1p(usage)."""
    if _NATIVE:
        return _native.compute_weight(impact_score, usage_count, success_count, failure_count)
    return _py_compute_weight(impact_score, usage_count, success_count, failure_count)


def batch_compute_weights(nodes: list[tuple[float, int, int, int]]) -> list[float]:
    """Compute weights for a batch of (impact, usage, success, failure) tuples."""
    if _NATIVE:
        return _native.batch_compute_weights(nodes)
    return _py_batch_compute_weights(nodes)


def argsort_by_weight(nodes: list[tuple[float, int, int, int]]) -> list[int]:
    """Return indices sorted by weight descending."""
    if _NATIVE:
        return _native.argsort_by_weight(nodes)
    return _py_argsort_by_weight(nodes)


def traverse_chain(
    id_to_parent: dict[str, Optional[str]],
    start_id: str,
    max_depth: int = 10,
) -> list[str]:
    """Walk the causal chain from *start_id* to the root.

    *id_to_parent* maps each node ID to its parent ID (or None for roots).
    Returns nodes ordered from oldest ancestor to *start_id*.
    Raises ValueError if a cycle is detected.
    """
    if _NATIVE:
        return _native.traverse_chain(id_to_parent, start_id, max_depth)
    return _py_traverse_chain(id_to_parent, start_id, max_depth)


def would_create_cycle(
    id_to_parent: dict[str, Optional[str]],
    child_id: str,
    parent_id: str,
) -> bool:
    """Return True if linking child → parent would create a cycle."""
    if _NATIVE:
        return _native.would_create_cycle(id_to_parent, child_id, parent_id)
    return _py_would_create_cycle(id_to_parent, child_id, parent_id)


def rank_by_similarity(
    query: list[float],
    embeddings: list[tuple[str, list[float]]],
    threshold: float = 0.7,
    top_k: int = 5,
) -> list[str]:
    """Return node IDs for the top-k embeddings most similar to *query*.

    *embeddings* is a list of (node_id, vector) pairs.
    Only nodes with cosine_similarity >= threshold are returned.
    """
    if _NATIVE:
        return _native.rank_by_similarity(query, embeddings, threshold, top_k)
    return _py_rank_by_similarity(query, embeddings, threshold, top_k)


def rank_blended(
    nodes: list[tuple[str, float, int, int, int, Optional[list[float]]]],
    query: list[float],
    sim_weight: float = 0.5,
    top_k: int = 10,
) -> list[str]:
    """Combined weight + similarity ranking.

    *nodes* is a list of (id, impact_score, usage_count, success_count, failure_count, embedding_or_None).
    *sim_weight* blends cosine similarity (1.0) vs weight score (0.0).
    Returns node IDs sorted by blended score descending.
    """
    if _NATIVE:
        return _native.rank_blended(nodes, query, sim_weight, top_k)
    return _py_rank_blended(nodes, query, sim_weight, top_k)


__all__ = [
    "is_native",
    "cosine_similarity",
    "batch_cosine_similarity",
    "compute_weight",
    "batch_compute_weights",
    "argsort_by_weight",
    "traverse_chain",
    "would_create_cycle",
    "rank_by_similarity",
    "rank_blended",
    "__version__",
]
