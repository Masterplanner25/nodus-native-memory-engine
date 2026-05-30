# nodus-native-memory-engine — Design Decision Log (Phase 0)

**Status:** Complete — 2026-05-29
**Decisions:** D1–D8 (all locked)

---

## D1 — No new Nodus opcodes

BYTECODE_VERSION stays at 4. This library is a pure performance layer — it
accelerates existing Python operations; it adds no language extensions, no new
builtins, and no VM changes.

## D2 — Python fallback is first-class, not an afterthought

Every public function has a complete pure-Python implementation in `__init__.py`.
When the Rust extension is absent, the library works identically (slower).
`is_native()` lets callers detect which path is active. Tests verify both paths
produce identical results within float precision.

**Why:** CI can run without Rust toolchain. Users who can't compile Rust get
working software. The library is useful before the wheel is built.

## D3 — Rust extension is an internal detail

The compiled extension is named `_core` (underscore prefix, per Python convention
for implementation modules). Callers import `nodus_native_memory_engine`, not
`nodus_native_memory_engine._core`. The `_core` name is not in `__all__` and
is subject to change.

## D4 — f64 throughout Rust code

All floating-point arithmetic uses `f64` (64-bit double), matching Python's native
`float` precision. Using f32 would introduce precision loss invisible to the caller.

## D5 — Maturin mixed mode (Python + Rust)

The package uses Maturin's mixed-module layout: `nodus_native_memory_engine/` is
the Python package root; `src/lib.rs` is the Rust extension compiled into
`nodus_native_memory_engine/_core.<platform>.so`.

`pyproject.toml` sets `module-name = "nodus_native_memory_engine._core"` so the
compiled extension lands in the right import path after `maturin develop` or
`pip install`.

## D6 — No numpy, no BLAS dependencies

The library uses only Rust's standard library for math (no ndarray, no BLAS bindings,
no SIMD intrinsics beyond what LLVM auto-vectorizes). This keeps the binary small
and the build reproducible on any Rust-capable platform.

**Rationale:** The primary win is eliminating Python loop overhead and GIL contention
on batch operations, not SIMD throughput. SIMD optimization (e.g., `pulp`, `wide`)
is a v0.2+ optimization deferred until benchmarks justify the added compile complexity.

## D7 — Rust extension result must match Python fallback within f64 precision

The invariant test verifies that for all implemented operations, `|native(x) - python(x)| < 1e-10`.
This is the contract that lets nodus-memory switch transparently between implementations.

## D8 — max_depth means max nodes (not max steps)

`traverse_chain(map, start, max_depth)` returns at most `max_depth` nodes, including
the start node. This matches nodus-memory's `recall_chain(key, max_depth=10)` semantics.
The Rust loop runs `max_depth - 1` times (start node already in chain; each iteration
adds one ancestor).
