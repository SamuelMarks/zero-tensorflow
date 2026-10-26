"""Unit tests for `test_quantization_stubs`."""

from __future__ import annotations

from pathlib import Path

from zero_tensorflow.quantization import experimental


def test_quantization(tmp_path: Path) -> None:
    """Verify `zero_tensorflow.quantization.experimental` classes and `quantize_saved_model`.

    Args:
        tmp_path: Temporary directory fixture.
    """
    comp = experimental.QuantizationComponentSpec(
        quantization_component=1, weight_type=2, tensor_type=3
    )
    assert comp.quantization_component == 1
    assert comp.weight_type == 2
    assert comp.tensor_type == 3

    method = experimental.QuantizationMethod(
        preset_method=experimental.QuantizationMethod.STATIC_RANGE
    )
    assert method.preset_method == 1

    opts = experimental.QuantizationOptions(quantization_method=method, op_set="XLA")
    assert opts.op_set == "XLA"
    assert opts.quantization_method is method

    saver = experimental.TfRecordRepresentativeDatasetSaver(
        path_map={"serving_default": "calib.tfrecord"}
    )
    assert saver.path_map["serving_default"] == "calib.tfrecord"

    unit_spec = experimental.UnitWiseQuantizationSpec(node_name="Conv2D", is_per_channel=False)
    assert unit_spec.node_name == "Conv2D"
    assert unit_spec.is_per_channel is False

    assert experimental.quantize_saved_model() is None
    src_dir = tmp_path / "src_model"
    src_dir.mkdir()
    (src_dir / "saved_model.json").write_text("{}", encoding="utf-8")
    dst_dir = tmp_path / "dst_model"
    res = experimental.quantize_saved_model(
        saved_model_path=str(src_dir),
        output_directory=str(dst_dir),
        quantization_options=opts,
    )
    assert res is not None
    assert res["op_set"] == "XLA"
    assert (dst_dir / "saved_model.json").is_file()
    res_default = experimental.quantize_saved_model(saved_model_path="nonexistent_dir")
    assert res_default is not None and res_default["op_set"] == "TFLITE"
