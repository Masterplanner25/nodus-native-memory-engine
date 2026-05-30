# nodus-native-memory-engine Design Doc 01 — Operations

**Doc:** 01-operations.md
**Status:** Complete — 2026-05-29
**Decisions grounded:** D4, D6, D7, D8
**Covers:** All 9 operations, their semantics, and their Rust implementations

---

## A — Similarity Operations

### A.1 `cosine_similarity(a, b) → f64`

Standard cosine similarity: `dot(a,b) / (|a| * |b|)`.

Edge cases:
- Zero vector: returns `0.0` (avoids division by zero)
- Length mismatch: raises `ValueError`
- Identical vectors: returns `1.0`
- All-zeros both: returns `0.0`

**Rust:** Pure f64 arithmetic via iterators. LLVM auto-vectorizes the dot product
and magnitude loops. No SIMD intrinsics required for correctness.

### A.2 `batch_cosine_similarity(query, matrix) → Vec<f64>`

Computes `cosine_similarity(query, row)` for each row in matrix.
Returns a list of scores in the same order as the rows.

**Performance gain over Python:** Eliminates the Python-level loop. For 1000 embeddings
of 1536 dimensions, replaces ~1.5M Python float operations with a single Rust call.

---

## B — Scoring Operations

### B.1 `compute_weight(impact, usage, success, failure) → f64`

Formula (identical to nodus-memory's `ScoreTracker.compute_weight`):
```
weight = impact * (1 + success_ratio) * ln(1 + usage)
```
Where `success_ratio = success / (success + failure)` or `0.0` if no feedback.

Edge case: `usage == 0` → `ln(1) = 0.0` → weight is always `0.0` regardless of other fields.

### B.2 `batch_compute_weights(nodes) → Vec<f64>`

Batch version: takes a list of `(impact, usage, success, failure)` tuples,
returns weights in the same order. No allocation of intermediate Python objects.

### B.3 `argsort_by_weight(nodes) → Vec<usize>`

Compute all weights, then return indices that would sort the list by weight
descending (highest weight first). Equivalent to `numpy.argsort(weights)[::-1]`
but without numpy.

Used internally by nodus-memory's `recall_all(sort_by="weight")`.

---

## C — Traversal Operations

### C.1 `traverse_chain(id_to_parent, start_id, max_depth) → Vec<String>`

Input: `HashMap<String, Option<String>>` mapping each node ID to its parent ID
(or `None` for root nodes).

Algorithm:
1. Start chain with `[start_id]`
2. Up to `max_depth - 1` times: look up current node's parent
3. If parent found and not yet seen: add to chain, advance current
4. If cycle (parent already seen): raise `ValueError`
5. Reverse chain (ancestors first) and return

**Semantics:** `max_depth` is the maximum number of nodes in the result including
`start_id`. A chain `[root, mid, leaf]` has depth 3.

### C.2 `would_create_cycle(id_to_parent, child_id, parent_id) → bool`

Pre-flight check before calling `link()`. Returns `True` if adding `child → parent`
would create a cycle.

Algorithm: Walk UP from `parent_id` through its ancestor chain. If we reach
`child_id`, a cycle would be created. Uses a `HashSet<String>` for O(n) detection.

---

## D — Combined Pipeline Operations

### D.1 `rank_by_similarity(query, embeddings, threshold, top_k) → Vec<String>`

Single Rust call that replaces the Python loop in `recall_similar()`:
1. Compute cosine similarity for each `(id, embedding)` pair
2. Filter by `threshold`
3. Sort descending by similarity
4. Return up to `top_k` node IDs

### D.2 `rank_blended(nodes, query, sim_weight, top_k) → Vec<String>`

Combined weight + similarity ranking:
- For each node: `score = sim_weight * cosine_sim + (1 - sim_weight) * weight`
- Where `weight = compute_weight(impact, usage, success, failure)`
- `sim_weight=0.0` → pure weight ranking
- `sim_weight=1.0` → pure similarity ranking
- `sim_weight=0.5` → balanced

Nodes without embeddings (`embedding=None`) get similarity score `0.0`.

---

## E — Integration with nodus-memory

When `nodus-native-memory-engine` is installed:

| nodus-memory function | Uses native |
|----------------------|-------------|
| `cosine_similarity()` in `embedding.py` | `_native_engine.cosine_similarity()` |
| `recall_similar()` | `_native_engine.rank_by_similarity()` (via cosine_similarity) |
| `ScoreTracker.compute_weight()` | `_native_engine.compute_weight()` |
| `ScoreTracker.batch_weights()` | `_native_engine.batch_compute_weights()` |
| `recall_all(sort_by="weight")` | calls `ScoreTracker.compute_weight()` per node (native) |

The integration is transparent: callers never know which implementation is active.

---

## F — Bytecode Impact

None. BYTECODE_VERSION stays at 4. (Decision D1)
No builtins are registered. No VM changes. No .nd language extensions.
This is a pure performance layer below the language abstraction.
