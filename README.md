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

v0.1.0 — published on [PyPI](https://pypi.org/project/nodus-native-memory-engine/).

## Pure Python fallback

The native Rust extension is optional. When not installed (or when the Rust
build fails), all operations fall back to pure Python implementations
automatically. Use `is_native()` to check which path is active:

```python
from nodus_native_memory_engine import is_native
print(is_native())   # True = Rust extension, False = pure Python
```

The fallback has the same API but is slower on large batches.

## Auto-detection by nodus-memory

Once installed, `nodus-memory` automatically detects and uses the native engine
for hot-path operations. No configuration needed.

## Build requirements (Rust extension)

- Rust 1.93.1+
- maturin 1.12.6+
- Python 3.11+

```bash
# Install with Rust extension (development)
VIRTUAL_ENV="C:/dev/Coding Language/.venv" maturin develop --release

# Install wheel only (pure Python fallback, no Rust needed)
pip install nodus-native-memory-engine
```

## Development

```bash
# Build and install Rust extension for development
maturin develop --release

# Run tests (works with or without Rust extension)
pytest tests/ -q
```

## License

MIT — see [LICENSE](LICENSE).
