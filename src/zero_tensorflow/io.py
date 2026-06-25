"""TensorFlow I/O module."""

from typing import Any, Optional

__all__ = [
    "read_file",
    "write_file",
    "decode_jpeg",
    "decode_png",
    "decode_image",
    "parse_tensor",
    "serialize_tensor",
    "parse_example",
    "parse_single_example",
]


def read_file(filename: str, name: Optional[str] = None) -> bytes:
    """
    Read the contents of a file.

    Args:
        filename (str): The path to the file.
        name (Optional[str]): A name for the operation (optional).

    Returns:
        bytes: The contents of the file.
    """
    with open(filename, "rb") as f:
        return f.read()


def write_file(filename: str, contents: bytes, name: Optional[str] = None) -> None:
    """
    Write contents to a file.

    Args:
        filename (str): The path to the file.
        contents (bytes): The contents to write.
        name (Optional[str]): A name for the operation (optional).
    """
    with open(filename, "wb") as f:
        f.write(contents)


def decode_jpeg(
    contents: Any,
    channels: int = 0,
    ratio: int = 1,
    fancy_upscaling: bool = True,
    try_recover_truncated: bool = False,
    acceptable_fraction: float = 1.0,
    dct_method: str = "",
    name: Optional[str] = None,
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
    raise NotImplementedError("Not implemented: tf.io.decode_jpeg")


def decode_png(
    contents: Any, channels: int = 0, dtype: Any = None, name: Optional[str] = None
) -> Any:
    """
    Decode a PNG-encoded image to a uint8 or uint16 tensor.

    Args:
        contents: A Tensor of type string. 0-D. The PNG-encoded image.
        channels: An optional int. Defaults to 0.
        dtype: An optional tf.DType from: tf.uint8, tf.uint16. Defaults to tf.uint8.
        name: A name for the operation (optional).

    Returns:
        A Tensor of type dtype.
    """
    raise NotImplementedError("Not implemented: tf.io.decode_png")


def decode_image(
    contents: Any,
    channels: Optional[int] = None,
    dtype: Any = None,
    name: Optional[str] = None,
    expand_animations: bool = True,
) -> Any:
    """
    Decode_bmp, decode_gif, decode_jpeg, and decode_png.

    Args:
        contents: A Tensor of type string. 0-D. The encoded image bytes.
        channels: An optional int. Defaults to 0.
        dtype: The optional type of the returned Tensor.
        name: A name for the operation (optional).
        expand_animations: An optional bool. Defaults to True.

    Returns:
        A Tensor of type dtype.
    """
    raise NotImplementedError("Not implemented: tf.io.decode_image")


def parse_tensor(serialized: Any, out_type: Any, name: Optional[str] = None) -> Any:
    """
    Transform a serialized tensorflow.TensorProto proto into a Tensor.

    Args:
        serialized: A Tensor of type string.
        out_type: A tf.DType.
        name: A name for the operation (optional).

    Returns:
        A Tensor of type out_type.
    """
    raise NotImplementedError("Not implemented: tf.io.parse_tensor")


def serialize_tensor(tensor: Any, name: Optional[str] = None) -> Any:
    """
    Transform a Tensor into a serialized TensorProto proto.

    Args:
        tensor: A Tensor.
        name: A name for the operation (optional).

    Returns:
        A Tensor of type string.
    """
    raise NotImplementedError("Not implemented: tf.io.serialize_tensor")


def parse_example(
    serialized: Any,
    features: Any,
    example_names: Optional[Any] = None,
    name: Optional[str] = None,
) -> Any:
    """
    Parse Example protos into a dict of tensors.

    Args:
        serialized: A tensor of strings.
        features: A dict mapping feature keys to FixedLenFeature or VarLenFeature values.
        example_names: A tensor of strings (optional).
        name: A name for the operation (optional).

    Returns:
        A dict mapping feature keys to Tensor and SparseTensor values.
    """
    raise NotImplementedError("Not implemented: tf.io.parse_example")


def parse_single_example(
    serialized: Any,
    features: Any,
    example_names: Optional[Any] = None,
    name: Optional[str] = None,
) -> Any:
    """
    Parse a single Example proto.

    Args:
        serialized: A scalar string Tensor, a single serialized Example.
        features: A dict mapping feature keys to FixedLenFeature or VarLenFeature values.
        example_names: A scalar string Tensor (optional).
        name: A name for the operation (optional).

    Returns:
        A dict mapping feature keys to Tensor and SparseTensor values.
    """
    raise NotImplementedError("Not implemented: tf.io.parse_single_example")


# Stubs from TODO_PLAN.md
from typing import Any


class FixedLenFeature:
    """Stub for FixedLenFeature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: FixedLenFeature")


class FixedLenSequenceFeature:
    """Stub for FixedLenSequenceFeature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: FixedLenSequenceFeature")


class RaggedFeature:
    """Stub for RaggedFeature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: RaggedFeature")


class SparseFeature:
    """Stub for SparseFeature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: SparseFeature")


class TFRecordOptions:
    """Stub for TFRecordOptions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: TFRecordOptions")


class TFRecordWriter:
    """Stub for TFRecordWriter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: TFRecordWriter")


class VarLenFeature:
    """Stub for VarLenFeature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: VarLenFeature")


def decode_and_crop_jpeg(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_and_crop_jpeg."""
    raise NotImplementedError("Not implemented: decode_and_crop_jpeg")


def decode_base64(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_base64."""
    raise NotImplementedError("Not implemented: decode_base64")


def decode_bmp(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_bmp."""
    raise NotImplementedError("Not implemented: decode_bmp")


def decode_compressed(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_compressed."""
    raise NotImplementedError("Not implemented: decode_compressed")


def decode_csv(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_csv."""
    raise NotImplementedError("Not implemented: decode_csv")


def decode_gif(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_gif."""
    raise NotImplementedError("Not implemented: decode_gif")


def decode_json_example(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_json_example."""
    raise NotImplementedError("Not implemented: decode_json_example")


def decode_proto(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_proto."""
    raise NotImplementedError("Not implemented: decode_proto")


def decode_raw(*args: Any, **kwargs: Any) -> None:
    """Stub for decode_raw."""
    raise NotImplementedError("Not implemented: decode_raw")


def deserialize_many_sparse(*args: Any, **kwargs: Any) -> None:
    """Stub for deserialize_many_sparse."""
    raise NotImplementedError("Not implemented: deserialize_many_sparse")


def encode_base64(*args: Any, **kwargs: Any) -> None:
    """Stub for encode_base64."""
    raise NotImplementedError("Not implemented: encode_base64")


def encode_jpeg(*args: Any, **kwargs: Any) -> None:
    """Stub for encode_jpeg."""
    raise NotImplementedError("Not implemented: encode_jpeg")


def encode_png(*args: Any, **kwargs: Any) -> None:
    """Stub for encode_png."""
    raise NotImplementedError("Not implemented: encode_png")


def encode_proto(*args: Any, **kwargs: Any) -> None:
    """Stub for encode_proto."""
    raise NotImplementedError("Not implemented: encode_proto")


def extract_jpeg_shape(*args: Any, **kwargs: Any) -> None:
    """Stub for extract_jpeg_shape."""
    raise NotImplementedError("Not implemented: extract_jpeg_shape")


class gfile:
    """Stub for gfile module."""

    class GFile:
        """Stub for GFile."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: GFile")

    @staticmethod
    def copy(*args: Any, **kwargs: Any) -> None:
        """Stub for copy."""
        raise NotImplementedError("Not implemented: copy")

    @staticmethod
    def exists(*args: Any, **kwargs: Any) -> None:
        """Stub for exists."""
        raise NotImplementedError("Not implemented: exists")

    @staticmethod
    def get_registered_schemes(*args: Any, **kwargs: Any) -> None:
        """Stub for get_registered_schemes."""
        raise NotImplementedError("Not implemented: get_registered_schemes")

    @staticmethod
    def glob(*args: Any, **kwargs: Any) -> None:
        """Stub for glob."""
        raise NotImplementedError("Not implemented: glob")

    @staticmethod
    def isdir(*args: Any, **kwargs: Any) -> None:
        """Stub for isdir."""
        raise NotImplementedError("Not implemented: isdir")

    @staticmethod
    def join(*args: Any, **kwargs: Any) -> None:
        """Stub for join."""
        raise NotImplementedError("Not implemented: join")

    @staticmethod
    def listdir(*args: Any, **kwargs: Any) -> None:
        """Stub for listdir."""
        raise NotImplementedError("Not implemented: listdir")

    @staticmethod
    def makedirs(*args: Any, **kwargs: Any) -> None:
        """Stub for makedirs."""
        raise NotImplementedError("Not implemented: makedirs")

    @staticmethod
    def mkdir(*args: Any, **kwargs: Any) -> None:
        """Stub for mkdir."""
        raise NotImplementedError("Not implemented: mkdir")

    @staticmethod
    def remove(*args: Any, **kwargs: Any) -> None:
        """Stub for remove."""
        raise NotImplementedError("Not implemented: remove")

    @staticmethod
    def rename(*args: Any, **kwargs: Any) -> None:
        """Stub for rename."""
        raise NotImplementedError("Not implemented: rename")

    @staticmethod
    def rmtree(*args: Any, **kwargs: Any) -> None:
        """Stub for rmtree."""
        raise NotImplementedError("Not implemented: rmtree")

    @staticmethod
    def stat(*args: Any, **kwargs: Any) -> None:
        """Stub for stat."""
        raise NotImplementedError("Not implemented: stat")

    @staticmethod
    def walk(*args: Any, **kwargs: Any) -> None:
        """Stub for walk."""
        raise NotImplementedError("Not implemented: walk")


def is_jpeg(*args: Any, **kwargs: Any) -> None:
    """Stub for is_jpeg."""
    raise NotImplementedError("Not implemented: is_jpeg")


def match_filenames_once(*args: Any, **kwargs: Any) -> None:
    """Stub for match_filenames_once."""
    raise NotImplementedError("Not implemented: match_filenames_once")


def matching_files(*args: Any, **kwargs: Any) -> None:
    """Stub for matching_files."""
    raise NotImplementedError("Not implemented: matching_files")


def parse_sequence_example(*args: Any, **kwargs: Any) -> None:
    """Stub for parse_sequence_example."""
    raise NotImplementedError("Not implemented: parse_sequence_example")


def parse_single_sequence_example(*args: Any, **kwargs: Any) -> None:
    """Stub for parse_single_sequence_example."""
    raise NotImplementedError("Not implemented: parse_single_sequence_example")


def serialize_many_sparse(*args: Any, **kwargs: Any) -> None:
    """Stub for serialize_many_sparse."""
    raise NotImplementedError("Not implemented: serialize_many_sparse")


def serialize_sparse(*args: Any, **kwargs: Any) -> None:
    """Stub for serialize_sparse."""
    raise NotImplementedError("Not implemented: serialize_sparse")


def write_graph(*args: Any, **kwargs: Any) -> None:
    """Stub for write_graph."""
    raise NotImplementedError("Not implemented: write_graph")
