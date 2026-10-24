"""Unit tests for `test_debugging`."""

from __future__ import annotations

from zero_tensorflow.debugging import experimental


def test_debugging() -> None:
    """Execute `test_debugging`."""
    assert experimental.disable_dump_debug_info() is None
    assert experimental.enable_dump_debug_info() is None
