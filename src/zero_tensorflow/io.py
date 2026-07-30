"""TensorFlow I/O module."""

from __future__ import annotations

from typing import Any

__all__ = [
    "decode_image",
    "decode_jpeg",
    "decode_png",
    "parse_example",
    "parse_single_example",
    "parse_tensor",
    "read_file",
    "serialize_tensor",
    "write_file",
]


def read_file(filename: str, name: str | None = None) -> bytes:
    """
    Read the contents of a file.

    Args:
        filename (str): The path to the file.
        name (Optional[str]): A name for the operation (optional).

    Returns:
        bytes: The contents of the file.
    """
    from ml_switcheroo_compiler.ops.io import read_file as _read_file

    return _read_file(filename)


def write_file(filename: str, contents: bytes, name: str | None = None) -> None:
    """
    Write contents to a file.

    Args:
        filename (str): The path to the file.
        contents (bytes): The contents to write.
        name (Optional[str]): A name for the operation (optional).
    """
    from ml_switcheroo_compiler.ops.io import write_file as _write_file

    _write_file(filename, contents)


def decode_jpeg(
    contents: Any,
    channels: int = 0,
    ratio: int = 1,
    fancy_upscaling: bool = True,
    try_recover_truncated: bool = False,
    acceptable_fraction: float = 1.0,
    dct_method: str = "",
    name: str | None = None,
) -> Any:
    """
    Decode a JPEG-encoded image to a uint8 tensor.

    Args:
        contents: A Tensor of type string. 0-D. The JPEG-encoded image.
        channels: An optional int. Defaults to 0.
        ratio: An optional int. Defaults to 1.
        fancy_upscaling: An optional bool. Defaults to True.
        try_recover_truncated: An optional bool. Defaults to False.
        acceptable_fraction: An optional float. Defaults to 1.
        dct_method: An optional string. Defaults to "".
        name: A name for the operation (optional).

    Returns:
        A Tensor of type uint8.
    """
    return None


def decode_png(contents: Any, channels: int = 0, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import decode_image as _decode_image

    return _decode_image(contents, channels)


def decode_image(contents: Any, channels: int = 0, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import decode_image as _decode_image

    return _decode_image(contents, channels)


def parse_tensor(serialized: Any, out_type: Any, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import parse_tensor as _parse_tensor

    return _parse_tensor(serialized, out_type)


def serialize_tensor(tensor: Any, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import serialize_tensor as _serialize_tensor

    return _serialize_tensor(tensor)


def parse_example(serialized: Any, features: Any, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import parse_example as _parse_example

    return _parse_example(serialized, features)


def parse_single_example(serialized: Any, features: Any, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import parse_example as _parse_example

    return _parse_example(serialized, features)


# Stubs from TODO_PLAN.md


class FixedLenFeature:
    """Stub for FixedLenFeature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class FixedLenSequenceFeature:
    """Stub for FixedLenSequenceFeature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class RaggedFeature:
    """Stub for RaggedFeature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class SparseFeature:
    """Stub for SparseFeature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class TFRecordOptions:
    """Stub for TFRecordOptions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class TFRecordWriter:
    """Stub for TFRecordWriter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class VarLenFeature:
    """Stub for VarLenFeature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def decode_and_crop_jpeg(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_and_crop_jpeg."""
    return


def decode_base64(*args: Any, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import decode_base64 as _fn

    return _fn(*args, **kwargs)


def decode_bmp(*args: Any, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import decode_bmp as _fn

    return _fn(*args, **kwargs)


def decode_compressed(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_compressed."""
    return


def decode_csv(records: Any, record_defaults: Any, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import decode_csv as _decode_csv

    return _decode_csv(records, record_defaults)


def decode_gif(*args: Any, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import decode_gif as _fn

    return _fn(*args, **kwargs)


def decode_json_example(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_json_example."""
    return


def decode_proto(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_proto."""
    return


def decode_raw(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_raw."""
    return


def deserialize_many_sparse(*args: Any, **kwargs: Any) -> None:
    """Stub for deserialize_many_sparse."""
    return


def encode_base64(*args: Any, **kwargs: Any) -> Any:
    from ml_switcheroo_compiler.ops.io import encode_base64 as _fn

    return _fn(*args, **kwargs)


def encode_jpeg(*args: Any, **kwargs: Any) -> Any:
    return None


def encode_png(*args: Any, **kwargs: Any) -> Any:
    return None


def encode_proto(*args: Any, **kwargs: Any) -> None:
    """Stub for encode_proto."""
    return


def extract_jpeg_shape(*args: Any, **kwargs: Any) -> None:
    """Stub for extract_jpeg_shape."""
    return


class gfile:
    """Stub for gfile module."""

    class GFile:
        """Stub for GFile."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    @staticmethod
    def copy(*args: Any, **kwargs: Any) -> Any:
        from ml_switcheroo_compiler.ops.io import gfile_copy as _fn

        return _fn(*args, **kwargs)

    @staticmethod
    def exists(*args: Any, **kwargs: Any) -> Any:
        import os

        return os.path.exists(*args, **kwargs)

    @staticmethod
    def get_registered_schemes(*args: Any, **kwargs: Any) -> None:
        """Stub for get_registered_schemes."""
        return

    @staticmethod
    def glob(*args: Any, **kwargs: Any) -> Any:
        from ml_switcheroo_compiler.ops.io import gfile_glob as _fn

        return _fn(*args, **kwargs)

    @staticmethod
    def isdir(*args: Any, **kwargs: Any) -> None:
        """Stub for isdir."""
        return

    @staticmethod
    def join(*args: Any, **kwargs: Any) -> None:
        """Stub for join."""
        return

    @staticmethod
    def listdir(*args: Any, **kwargs: Any) -> None:
        """Stub for listdir."""
        return

    @staticmethod
    def makedirs(*args: Any, **kwargs: Any) -> Any:
        from ml_switcheroo_compiler.ops.io import gfile_makedirs as _fn

        return _fn(*args, **kwargs)

    @staticmethod
    def mkdir(*args: Any, **kwargs: Any) -> None:
        """Stub for mkdir."""
        return

    @staticmethod
    def remove(*args: Any, **kwargs: Any) -> None:
        """Stub for remove."""
        return

    @staticmethod
    def rename(*args: Any, **kwargs: Any) -> None:
        """Stub for rename."""
        return

    @staticmethod
    def rmtree(*args: Any, **kwargs: Any) -> None:
        """Stub for rmtree."""
        return

    @staticmethod
    def stat(*args: Any, **kwargs: Any) -> Any:
        from ml_switcheroo_compiler.ops.io import gfile_stat as _fn

        return _fn(*args, **kwargs)

    @staticmethod
    def walk(*args: Any, **kwargs: Any) -> None:
        """Stub for walk."""
        return


def is_jpeg(*args: Any, **kwargs: Any) -> None:
    """Stub for is_jpeg."""
    return


def match_filenames_once(*args: Any, **kwargs: Any) -> None:
    """Stub for match_filenames_once."""
    return


def matching_files(*args: Any, **kwargs: Any) -> None:
    """Stub for matching_files."""
    return


def parse_sequence_example(*args: Any, **kwargs: Any) -> None:
    """Stub for parse_sequence_example."""
    return


def parse_single_sequence_example(*args: Any, **kwargs: Any) -> None:
    """Stub for parse_single_sequence_example."""
    return


def serialize_many_sparse(*args: Any, **kwargs: Any) -> None:
    """Stub for serialize_many_sparse."""
    return


def serialize_sparse(*args: Any, **kwargs: Any) -> None:
    """Stub for serialize_sparse."""
    return


def write_graph(*args: Any, **kwargs: Any) -> None:
    """Stub for write_graph."""
    return
