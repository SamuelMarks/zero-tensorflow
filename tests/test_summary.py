"""Unit tests for `test_summary`."""

from __future__ import annotations

from zero_tensorflow.summary import experimental, image


def test_summary() -> None:
    """Execute `test_summary`."""
    experimental.set_step(42)
    assert experimental.get_step() == 42
    with experimental.summary_scope("test") as (name, val):
        assert name == "summary_scope"
        assert val is None
    assert experimental.write_raw_pb(b"data") is None
    image("test_image", None)
