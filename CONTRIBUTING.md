# Contributing to nodus-native-memory-engine

## Setup

This package has both a Rust extension (`src/lib.rs`) and a pure Python
fallback (`nodus_native_memory_engine/__init__.py`). Tests run against
whichever is installed.

### With Rust extension (full dev)

```bash
git clone https://github.com/Masterplanner25/nodus-native-memory-engine.git
cd nodus-native-memory-engine

# Build Rust extension
maturin develop --release

# Run tests
pytest tests/ -q
```

Requires: Rust 1.93.1+, maturin 1.12.6+.

### Pure Python only (no Rust build)

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

Tests pass against the pure Python fallback — `is_native()` returns `False`.

## Code style

- Python 3.11+
- Rust: edition 2021, PyO3 0.22.6
- Every operation must have both a Rust implementation (`src/lib.rs`) and a
  pure Python fallback (`nodus_native_memory_engine/__init__.py`)
- `is_native()` must always be callable regardless of which path is active

## Test structure

| File | Coverage |
|---|---|
| `test_similarity.py` | cosine_similarity, batch_cosine_similarity |
| `test_scoring.py` | compute_weight, batch_compute_weights, argsort_by_weight |
| `test_traversal.py` | traverse_chain, would_create_cycle |
| `test_ranking.py` | rank_by_similarity, rank_blended |
| `test_invariants.py` | API contract assertions; is_native() behaviour |

## Submitting changes

1. Fork the repo and create a branch from `main`
2. Add matching tests for any new operation
3. Ensure both Rust and pure Python paths are implemented
4. Ensure `pytest tests/ -q` passes
5. Open a pull request with a description of what changes and why
