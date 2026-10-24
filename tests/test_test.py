"""Unit tests for `test_test`."""

from __future__ import annotations

from zero_tensorflow.test import experimental


def test_test() -> None:
    """Execute `test_test`."""
    assert experimental.sync_devices() is None
