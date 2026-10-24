"""Unit tests for `test_saved_model`."""

from __future__ import annotations

from zero_tensorflow._typing import TensorOperand
from zero_tensorflow.saved_model import (
    Asset,
    LoadOptions,
    SaveOptions,
    contains_saved_model,
    experimental,
    load,
    save,
)


def test_saved_model() -> None:
    """Execute `test_saved_model`."""
    assert Asset() is not None
    assert LoadOptions() is not None
    assert SaveOptions() is not None
    contains_saved_model()
    assert experimental.Fingerprint() is not None
    assert experimental.TrackableResource() is not None
    assert experimental.VariablePolicy() is not None
    assert experimental.read_fingerprint() is None
    save(None, "path")

    def dummy_call(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
        """Execute `dummy_call`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorOperand: Computed result.
        """
        del args, kwargs
        return None

    save(dummy_call, "path")
    load("invalid_path")
