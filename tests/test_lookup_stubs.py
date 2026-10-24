"""Unit tests for `test_lookup_stubs`."""

from __future__ import annotations

from zero_tensorflow.lookup import experimental


def test_lookup() -> None:
    """Execute `test_lookup`."""
    assert experimental.DenseHashTable() is not None
    assert experimental.MutableHashTable() is not None
