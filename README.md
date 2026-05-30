# nodus-native-memory-engine

Rust-accelerated memory scoring, similarity, and traversal for Nodus agents.

Provides native implementations of the hot-path operations from `nodus-memory`:
cosine similarity, weight scoring, causal chain traversal, and combined ranking.

## Installation

```
pip install nodus-native-memory-engine
```

After installation, nodus-memory automatically detects and uses the native engine.

## Operations

| Function | Description |
|----------|-------------|
| `cosine_similarity(a, b)` | Single-pair cosine similarity |
| `batch_cosine_similarity(query, matrix)` | One query vs many embeddings |
| `compute_weight(impact, usage, success, fail)` | Memory node ranking weight |
| `batch_compute_weights(nodes)` | Batch weight computation |
| `argsort_by_weight(nodes)` | Sort indices by weight descending |
| `traverse_chain(id_to_parent, start, depth)` | Causal chain walk + cycle detection |
| `would_create_cycle(id_to_parent, child, parent)` | Cycle pre-check |
| `rank_by_similarity(query, embeddings, ...)` | Top-k by cosine similarity |
| `rank_blended(nodes, query, sim_weight, top_k)` | Combined weight + similarity ranking |
| `is_native()` | True if Rust extension is loaded |

All functions have pure-Python fallbacks — the package works even without the compiled extension.

## Status

v0.1.0 — PREPARED, NOT RELEASED.
