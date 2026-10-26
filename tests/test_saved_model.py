"""Unit tests for `test_saved_model`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import zero_tensorflow.saved_model as sm_mod
from zero_tensorflow import Tensor, Variable
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


class _DummyModel:
    """Trackable container for testing `saved_model.save` and `saved_model.load` disk persistence."""

    def __init__(self) -> None:
        """Initialize `_DummyModel` with two `Variable`s."""
        self._vars = [Variable([1.0, 2.0]), Variable(3.0)]

    @property
    def variables(self) -> Sequence[Variable]:
        """Return tracked variables.

        Returns:
            Sequence[Variable]: Tracked variables.
        """
        return self._vars


def test_saved_model(tmp_path: Path) -> None:
    """Verify `saved_model` disk export, fingerprint reading, and in-memory fallback.

    Args:
        tmp_path: Temporary directory fixture.
    """
    assert Asset("a.txt").asset_path == "a.txt"
    assert LoadOptions(allow_partial_checkpoint=True).allow_partial_checkpoint is True
    assert SaveOptions(save_debug_info=True).save_debug_info is True
    assert contains_saved_model() is False
    assert experimental.Fingerprint() is not None
    assert experimental.TrackableResource(device="/cpu:0")._resource_device == "/cpu:0"
    assert experimental.VariablePolicy("NONE").value == "NONE"
    assert experimental.read_fingerprint() is None
    save(None, "path")
    assert contains_saved_model("path") is True
    assert experimental.read_fingerprint("path") is not None

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
    assert load("path") is dummy_call
    assert load("invalid_path") is None

    # Disk persistence in tmp_path
    export_dir = str(tmp_path / "exported_model")
    model = _DummyModel()
    save(model, export_dir)
    assert (tmp_path / "exported_model" / "saved_model.pb").is_file()
    assert (tmp_path / "exported_model" / "fingerprint.pb").is_file()
    assert (tmp_path / "exported_model" / "variables").is_dir()
    assert (tmp_path / "exported_model" / "assets").is_dir()

    sm_mod._SAVED_MODEL_STORE.pop(export_dir, None)
    sm_mod._FINGERPRINT_STORE.pop(export_dir, None)
    assert contains_saved_model(export_dir) is True
    loaded_from_disk = load(export_dir)
    assert loaded_from_disk is not None and callable(loaded_from_disk)
    out0 = loaded_from_disk()
    assert isinstance(out0, Tensor) and out0.shape == (3,)
    out1 = loaded_from_disk(7.0)
    assert isinstance(out1, Tensor) and out1.numpy() == 7.0
    fp = experimental.read_fingerprint(export_dir)
    assert fp is not None and fp.saved_model_checksum > 0
