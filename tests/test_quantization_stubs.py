"""Unit tests for `test_quantization_stubs`."""

from __future__ import annotations

from zero_tensorflow.quantization import experimental


def test_quantization() -> None:
    """Execute `test_quantization`."""
    assert experimental.QuantizationComponentSpec() is not None
    assert experimental.QuantizationMethod() is not None
    assert experimental.QuantizationOptions() is not None
    assert experimental.TfRecordRepresentativeDatasetSaver() is not None
    assert experimental.UnitWiseQuantizationSpec() is not None
    assert experimental.quantize_saved_model() is None
