"""Unit tests for `test_autograph_stubs`."""

from __future__ import annotations

from zero_tensorflow.autograph import experimental


def test_autograph() -> None:
    """Execute `test_autograph`."""
    assert experimental.Feature() is not None
    assert experimental.do_not_convert() is not None
    assert experimental.set_loop_options() is None
