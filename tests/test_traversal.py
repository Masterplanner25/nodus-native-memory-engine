"""Causal chain traversal and cycle detection."""
from __future__ import annotations

import pytest
from nodus_native_memory_engine import (
    traverse_chain,
    would_create_cycle,
    _py_traverse_chain,
    _py_would_create_cycle,
)


def _chain_map(*ids):
    """Build id_to_parent for a linear chain: ids[0] is root, ids[-1] is leaf."""
    m = {}
    for i, node_id in enumerate(ids):
        m[node_id] = ids[i - 1] if i > 0 else None
    return m


class TestTraverseChain:
    def test_single_node_no_parent(self):
        result = traverse_chain({"a": None}, "a", 10)
        assert result == ["a"]

    def test_linear_chain_two_nodes(self):
        m = {"root": None, "child": "root"}
        result = traverse_chain(m, "child", 10)
        assert result == ["root", "child"]

    def test_linear_chain_three_nodes(self):
        m = _chain_map("r", "m", "l")
        result = traverse_chain(m, "l", 10)
        assert result == ["r", "m", "l"]

    def test_max_depth_truncates(self):
        ids = [str(i) for i in range(10)]
        m = _chain_map(*ids)
        result = traverse_chain(m, ids[-1], 3)
        assert len(result) <= 3

    def test_start_not_in_map_returns_just_start(self):
        result = traverse_chain({}, "orphan", 10)
        assert result == ["orphan"]

    def test_root_returns_just_root(self):
        m = {"root": None, "child": "root"}
        result = traverse_chain(m, "root", 10)
        assert result == ["root"]

    def test_cycle_raises_value_error(self):
        # Manually create a cycle: a→b, b→a
        m = {"a": "b", "b": "a"}
        with pytest.raises((ValueError, Exception)):
            traverse_chain(m, "a", 10)


class TestWouldCreateCycle:
    def test_simple_no_cycle(self):
        m = {"root": None, "child": "root"}
        assert not would_create_cycle(m, "grandchild", "child")

    def test_direct_cycle_detected(self):
        # a→b; linking b→a would create a↔b cycle
        m = {"a": "b", "b": None}
        assert would_create_cycle(m, "b", "a")

    def test_indirect_cycle_detected(self):
        # b's parent=a, c's parent=b (chain: a is root, b is mid, c is leaf).
        # Linking a→c (setting a's parent to c) creates cycle: a←b←c←a.
        m = {"a": None, "b": "a", "c": "b"}
        assert would_create_cycle(m, "a", "c")

    def test_no_cycle_when_linear(self):
        m = {"a": None, "b": "a", "c": "b"}
        assert not would_create_cycle(m, "d", "c")

    def test_empty_map_no_cycle(self):
        assert not would_create_cycle({}, "child", "parent")

    def test_self_link_detected(self):
        m = {"a": None}
        assert would_create_cycle(m, "a", "a")


class TestFallbackMatchesNative:
    def test_traverse_chain_matches(self):
        m = _chain_map("r", "m", "l")
        native = traverse_chain(m, "l", 10)
        python = _py_traverse_chain(m, "l", 10)
        assert native == python

    def test_would_create_cycle_matches(self):
        m = {"a": None, "b": "a", "c": "b"}
        cases = [
            ("d", "c"),   # no cycle
            ("c", "a"),   # cycle: a→b→c→a
        ]
        for child, parent in cases:
            native = would_create_cycle(m, child, parent)
            python = _py_would_create_cycle(m, child, parent)
            assert native == python, f"mismatch for ({child}, {parent}): native={native}, python={python}"
