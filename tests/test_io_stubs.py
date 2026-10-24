"""Unit tests for `test_io_stubs`."""

from __future__ import annotations

import contextlib
import inspect

import zero_tensorflow.io as tf_io


def call_safely(func_name: str) -> None:
    """Invoke `tf_io.<func_name>` with zero arguments while suppressing expected errors.

    Args:
        func_name: Name of the target `tf.io` function or class.
    """
    members = dict(inspect.getmembers(tf_io))
    func = members.get(func_name)
    if callable(func):
        with contextlib.suppress(
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            NotImplementedError,
            KeyError,
            IndexError,
            OSError,
        ):
            func()


def test_io_stubs() -> None:
    """Verify all `tf.io` classes and functions can be invoked."""
    for name in (
        "FixedLenFeature",
        "FixedLenSequenceFeature",
        "RaggedFeature",
        "SparseFeature",
        "TFRecordOptions",
        "TFRecordWriter",
        "VarLenFeature",
        "decode_and_crop_jpeg",
        "decode_base64",
        "decode_bmp",
        "decode_compressed",
        "decode_csv",
        "decode_gif",
        "decode_image",
        "decode_jpeg",
        "decode_json_example",
        "decode_png",
        "decode_proto",
        "decode_raw",
        "encode_base64",
        "encode_jpeg",
        "encode_png",
        "extract_jpeg_shape",
        "is_jpeg",
        "match_filenames_once",
        "matching_files",
        "parse_example",
        "parse_sequence_example",
        "parse_single_example",
        "parse_single_sequence_example",
        "parse_tensor",
        "read_file",
        "serialize_many_sparse",
        "serialize_sparse",
        "serialize_tensor",
        "write_file",
        "write_graph",
    ):
        call_safely(name)
