# Changelog

Format: [Keep a Changelog](https://keepachangelog.com)
Versioning: [Semantic Versioning](https://semver.org)

## [Unreleased]

## [0.1.0] — PREPARED, NOT RELEASED

> **Coordinated launch:** nodus-native-memory-engine v0.1.0 is prepared but
> not published. Publication follows the three-artifact set (nodus-lang 4.0.0,
> nodus-mcp 0.1.0, nodus-a2a 0.1.0) and nodus-memory 0.1.0.

### Added

**Rust extension (PyO3 / Maturin)**

- `cosine_similarity(a, b)` — single-pair cosine similarity with zero-vector safety
- `batch_cosine_similarity(query, matrix)` — one query vs N embeddings in one Rust call
- `compute_weight(impact, usage, success, failure)` — memory node ranking weight
- `batch_compute_weights(nodes)` — batch weight computation
- `argsort_by_weight(nodes)` — indices sorted by weight descending
- `traverse_chain(id_to_parent, start, max_depth)` — causal chain walk + cycle detection
- `would_create_cycle(id_to_parent, child, parent)` — O(n) cycle pre-check
- `rank_by_similarity(query, embeddings, threshold, top_k)` — top-k by cosine similarity
- `rank_blended(nodes, query, sim_weight, top_k)` — combined weight + similarity pipeline

**Python layer**

- Pure-Python fallback for every function — library works without the compiled extension
- `is_native()` — detect whether Rust extension is loaded
- All functions route to Rust or Python transparently

**nodus-memory integration**

- `nodus_memory.embedding.cosine_similarity()` auto-uses native when available
- `nodus_memory.scoring.ScoreTracker.compute_weight()` auto-uses native when available
- `nodus_memory.scoring.ScoreTracker.batch_weights()` added (native batch path)
- Integration is transparent: nodus-memory tests pass unchanged

**Quality**

- 76 tests; native and fallback results verified to match within 1e-10
- 8 design decisions locked (00-decisions.md)
- 2 design docs: 00-decisions, 01-operations
- BYTECODE_VERSION unchanged (4)

### Design Decisions

- **D1** — No new Nodus opcodes; BYTECODE_VERSION stays 4
- **D2** — Python fallback is first-class; library works without Rust
- **D3** — Rust extension is internal (`_core`); public API is Python
- **D4** — f64 throughout Rust (matches Python float precision)
- **D5** — Maturin mixed mode (Python package + compiled `_core` extension)
- **D6** — No numpy/BLAS; pure Rust stdlib math only
- **D7** — Native results must match Python within 1e-10
- **D8** — max_depth = max number of nodes (not steps)
