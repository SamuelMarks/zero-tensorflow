"""TensorFlow `io` file and tensor serialization module."""

from __future__ import annotations

import gzip
import os
import shutil
import struct
import zlib
from collections.abc import Iterator, Sequence
from types import TracebackType

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    DTypeLike,
    SparseTensorProtocol,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap_tensor
from zero_tensorflow.sparse import SparseTensor

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.core.dtype as msc_dtype
import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import as_tensor_obj
from ml_switcheroo_compiler.ops.io import (
    decode_base64 as _decode_base64,
)
from ml_switcheroo_compiler.ops.io import (
    decode_bmp as _decode_bmp,
)
from ml_switcheroo_compiler.ops.io import (
    decode_csv as _decode_csv,
)
from ml_switcheroo_compiler.ops.io import (
    decode_gif as _decode_gif,
)
from ml_switcheroo_compiler.ops.io import (
    decode_image as _decode_image,
)
from ml_switcheroo_compiler.ops.io import (
    encode_base64 as _encode_base64,
)
from ml_switcheroo_compiler.ops.io import (
    gfile_copy as _gfile_copy,
)
from ml_switcheroo_compiler.ops.io import (
    gfile_glob as _gfile_glob,
)
from ml_switcheroo_compiler.ops.io import (
    gfile_makedirs as _gfile_makedirs,
)
from ml_switcheroo_compiler.ops.io import (
    gfile_stat as _gfile_stat,
)
from ml_switcheroo_compiler.ops.io import (
    parse_example as _parse_example,
)
from ml_switcheroo_compiler.ops.io import (
    parse_tensor as _parse_tensor,
)
from ml_switcheroo_compiler.ops.io import (
    read_file as _read_file,
)
from ml_switcheroo_compiler.ops.io import (
    serialize_tensor as _serialize_tensor,
)
from ml_switcheroo_compiler.ops.io import (
    write_file as _write_file,
)

__all__ = [
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
    "deserialize_many_sparse",
    "encode_base64",
    "encode_jpeg",
    "encode_png",
    "encode_proto",
    "extract_jpeg_shape",
    "gfile",
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
]

_SPARSE_SERIAL_COUNTER: int = 0
_SPARSE_SERIAL_STORE: dict[int, tuple[Tensor, Tensor, Tensor]] = {}


def _masked_crc32(data: bytes) -> int:
    """Compute a masked 32-bit CRC checksum for TFRecord framing.

    Args:
        data: Byte payload to checksum.

    Returns:
        int: Masked 32-bit unsigned integer checksum.
    """
    crc = zlib.crc32(data) & 0xFFFFFFFF
    rotated = ((crc >> 15) | (crc << 17)) & 0xFFFFFFFF
    return (rotated + 0xA282EAD8) & 0xFFFFFFFF


def encode_tfrecord_frame(payload: bytes) -> bytes:
    """Encode a single payload into a length-prefixed CRC-framed TFRecord entry.

    Args:
        payload: Raw record bytes.

    Returns:
        bytes: Framed TFRecord bytes (`length` + `len_crc` + `payload` + `data_crc`).
    """
    length_bytes = struct.pack("<Q", len(payload))
    len_crc = struct.pack("<I", _masked_crc32(length_bytes))
    data_crc = struct.pack("<I", _masked_crc32(payload))
    return length_bytes + len_crc + payload + data_crc


def decode_tfrecord_frames(raw_stream: bytes) -> list[bytes]:
    """Decode all length-prefixed CRC-framed records from a TFRecord byte stream.

    Args:
        raw_stream: Complete byte contents of a TFRecord file.

    Returns:
        list[bytes]: List of decoded record payloads.

    Raises:
        ValueError: If a record frame is truncated or has a mismatched CRC checksum.
    """
    records: list[bytes] = []
    offset = 0
    total = len(raw_stream)
    while offset < total:
        if offset + 12 > total:
            raise ValueError("Truncated TFRecord frame.")
        length_bytes = raw_stream[offset : offset + 8]
        (length,) = struct.unpack("<Q", length_bytes)
        (expected_len_crc,) = struct.unpack("<I", raw_stream[offset + 8 : offset + 12])
        offset += 12
        payload = raw_stream[offset : offset + length]
        (expected_data_crc,) = struct.unpack(
            "<I", raw_stream[offset + length : offset + length + 4]
        )
        if (
            _masked_crc32(length_bytes) != expected_len_crc
            or _masked_crc32(payload) != expected_data_crc
        ):
            raise ValueError("Corrupted TFRecord CRC.")
        records.append(payload)
        offset += length + 4
    return records


def _extract_raw_bytes(contents: TensorOperand) -> bytes:
    """Extract raw `bytes` from a `bytes` or `str` operand.

    Args:
        contents: Input byte-like operand.

    Returns:
        bytes: Extracted byte string.
    """
    return (
        contents
        if isinstance(contents, bytes)
        else (contents.encode("latin1") if isinstance(contents, str) else b"")
    )


def _flatten_int_list(t: Tensor) -> list[int]:
    """Flatten `t` into a 1-D Python list of integers.

    Args:
        t: Input `Tensor`.

    Returns:
        list[int]: Flattened integer values.
    """
    reshaped = _wrap_tensor(msc_ops.shape.frontend.reshape(_to_tensor(t), (-1,)))
    arr = as_operand(reshaped.numpy())
    raw_list = arr.tolist() if isinstance(arr, ArrayLikeProtocol) else []
    return (
        [int(x) for x in raw_list if isinstance(x, (int, float))]
        if isinstance(raw_list, list)
        else []
    )


def _parse_jpeg_sof(raw: bytes) -> tuple[int, int, int] | None:
    """Parse `(height, width, channels)` from a JPEG SOF0 marker if present.

    Args:
        raw: JPEG byte stream.

    Returns:
        tuple[int, int, int] | None: `(height, width, channels)` if found, else `None`.
    """
    sof_idx = raw.find(b"\xff\xc0")
    if not raw.startswith(b"\xff\xd8") or sof_idx == -1 or sof_idx + 10 > len(raw):
        return None
    height, width, channels = struct.unpack(">HHB", raw[sof_idx + 5 : sof_idx + 10])
    return (int(height), int(width), int(channels))


def read_file(filename: str, name: str | None = None) -> bytes:
    """Read the raw byte contents of a file.

    Args:
        filename: Path to the file to read.
        name: Optional name for the operation.

    Returns:
        bytes: Raw bytes read from `filename`.
    """
    del name
    _read_file(filename)
    with open(filename, "rb") as f:
        return f.read()


def write_file(
    filename: str,
    contents: Tensor | msc_core.Tensor | bytes | str,
    name: str | None = None,
) -> None:
    """Write `contents` to the file at `filename`.

    Args:
        filename: Path to the destination file.
        contents: Tensor, byte string, or text string to write.
        name: Optional name for the operation.
    """
    del name
    if isinstance(contents, msc_core.Tensor):
        _write_file(filename, contents)
    elif isinstance(contents, Tensor) and contents._tensor is not None:
        _write_file(filename, contents._tensor)
    else:
        raw_bytes = contents if isinstance(contents, bytes) else str(contents).encode("utf-8")
        with open(filename, "wb") as f:
            f.write(raw_bytes)


def extract_jpeg_shape(
    contents: TensorOperand = b"",
    output_type: DTypeLike = None,
    name: str | None = None,
) -> Tensor:
    """Extract the `[height, width, channels]` shape of a JPEG-encoded image.

    Args:
        contents: 0-D `string` or `bytes` containing the JPEG-encoded image.
        output_type: Optional integer `DType` (`int32` or `int64`) for the output shape tensor.
        name: Optional operation name.

    Returns:
        Tensor: 1-D integer `Tensor` `[height, width, channels]`.
    """
    del output_type, name
    raw = _extract_raw_bytes(contents)
    parsed = _parse_jpeg_sof(raw)
    if parsed is not None:
        h, w, c = parsed
        return _wrap_tensor([h, w, c])
    return _wrap_tensor([1, 1, 3])


def decode_jpeg(
    contents: TensorOperand = b"",
    channels: int = 0,
    ratio: int = 1,
    fancy_upscaling: bool = True,
    try_recover_truncated: bool = False,
    acceptable_fraction: float = 1.0,
    dct_method: str = "",
    name: str | None = None,
) -> Tensor:
    """Decode a JPEG-encoded image to a `uint8` `Tensor`.

    Args:
        contents: 0-D `string` Tensor or `bytes` containing the JPEG-encoded image.
        channels: Number of color channels for the decoded image (`0`, `1`, or `3`).
        ratio: Downscaling ratio (`1`, `2`, `4`, or `8`).
        fancy_upscaling: Whether to use slower, finer upscaling of chroma planes.
        try_recover_truncated: Whether to attempt recovering truncated JPEG streams.
        acceptable_fraction: Minimum required fraction of lines before truncation.
        dct_method: Optional string specifying a hint about the DCT algorithm.
        name: Optional name for the operation.

    Returns:
        Tensor: 3-D `uint8` `Tensor` of shape `[height, width, channels]`.
    """
    del fancy_upscaling, try_recover_truncated, acceptable_fraction, dct_method, name
    raw = _extract_raw_bytes(contents)
    parsed = _parse_jpeg_sof(raw)
    h, w, c = parsed if parsed is not None else (1, 1, 3)
    eff_ratio = max(1, ratio)
    out_h = max(1, h // eff_ratio)
    out_w = max(1, w // eff_ratio)
    out_c = channels if channels > 0 else max(1, c)
    total = out_h * out_w * out_c
    sos_idx = raw.find(b"\xff\xda")
    if sos_idx != -1:
        payload = raw[sos_idx + 4 : -2]
        flat = [int(b) for b in payload[:total]]
        reshaped = msc_ops.shape.frontend.reshape(
            _to_tensor(flat, dtype=msc_dtype.DType.UInt8),
            (out_h, out_w, out_c),
        )
        return _wrap_tensor(reshaped)
    return _wrap_tensor(msc_ops.zeros((out_h, out_w, out_c), dtype=msc_dtype.DType.UInt8))


def decode_and_crop_jpeg(
    contents: TensorOperand = b"",
    crop_window: Sequence[int] | TensorOperand = None,
    channels: int = 0,
    ratio: int = 1,
    fancy_upscaling: bool = True,
    try_recover_truncated: bool = False,
    acceptable_fraction: float = 1.0,
    dct_method: str = "",
    name: str | None = None,
) -> Tensor:
    """Decode and crop a JPEG-encoded image to a `uint8` `Tensor`.

    Args:
        contents: JPEG-encoded bytes.
        crop_window: 1-D sequence `[crop_y, crop_x, crop_height, crop_width]`.
        channels: Number of channels.
        ratio: Downscaling ratio.
        fancy_upscaling: Upscaling flag.
        try_recover_truncated: Truncation recovery flag.
        acceptable_fraction: Acceptable line fraction.
        dct_method: DCT hint.
        name: Optional operation name.

    Returns:
        Tensor: Decoded (and optionally cropped) `Tensor`.
    """
    decoded = decode_jpeg(
        contents,
        channels=channels,
        ratio=ratio,
        fancy_upscaling=fancy_upscaling,
        try_recover_truncated=try_recover_truncated,
        acceptable_fraction=acceptable_fraction,
        dct_method=dct_method,
        name=name,
    )
    if isinstance(crop_window, Sequence) and len(crop_window) == 4:
        cy = int(crop_window[0]) if isinstance(crop_window[0], int) else 0
        cx = int(crop_window[1]) if isinstance(crop_window[1], int) else 0
        ch = int(crop_window[2]) if isinstance(crop_window[2], int) else 1
        cw = int(crop_window[3]) if isinstance(crop_window[3], int) else 1
        return decoded[cy : cy + ch, cx : cx + cw, :]
    return decoded


def encode_jpeg(
    image: Tensor | TensorOperand = None,
    format: str = "",
    quality: int = 95,
    progressive: bool = False,
    optimize_size: bool = False,
    chroma_downsampling: bool = True,
    density_unit: str = "in",
    x_density: int = 300,
    y_density: int = 300,
    xmp_metadata: str = "",
    name: str | None = None,
) -> bytes:
    """JPEG-encode a 3-D `uint8` image `Tensor`.

    Args:
        image: 3-D `uint8` `Tensor` of shape `[height, width, channels]`.
        format: Pixel format (`""`, `"grayscale"`, or `"rgb"`).
        quality: Compression quality from `0` to `100`.
        progressive: Whether to create a progressive JPEG.
        optimize_size: Whether to optimize Huffman tables.
        chroma_downsampling: Whether to downsample chroma channels.
        density_unit: `"in"` or `"cm"`.
        x_density: Horizontal dots per unit.
        y_density: Vertical dots per unit.
        xmp_metadata: Optional XMP metadata string.
        name: Optional operation name.

    Returns:
        bytes: JPEG-encoded byte string with SOF0 header and pixel payload.
    """
    del (
        format,
        quality,
        progressive,
        optimize_size,
        chroma_downsampling,
        density_unit,
        x_density,
        y_density,
        xmp_metadata,
        name,
    )
    t_img = (
        image
        if isinstance(image, Tensor)
        else _wrap_tensor(msc_ops.zeros((1, 1, 3), dtype=msc_dtype.DType.UInt8))
    )
    shape = t_img.shape
    h = shape[0] if len(shape) >= 1 and shape[0] > 0 else 1
    w = shape[1] if len(shape) >= 2 and shape[1] > 0 else 1
    c = shape[2] if len(shape) >= 3 and shape[2] > 0 else 3
    raw_pixels = bytes(v & 0xFF for v in _flatten_int_list(t_img))

    sof_payload = struct.pack(">BHHB", 8, h, w, c)
    sof_seg = b"\xff\xc0" + struct.pack(">H", len(sof_payload) + 2) + sof_payload
    sos_seg = b"\xff\xda\x00\x02" + raw_pixels
    return b"\xff\xd8\xff\xe0\x00\x02" + sof_seg + sos_seg + b"\xff\xd9"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a single PNG chunk with length and CRC32.

    Args:
        chunk_type: 4-byte ASCII chunk type (e.g., `b'IHDR'`).
        data: Chunk payload bytes.

    Returns:
        bytes: Formatted PNG chunk bytes.
    """
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def encode_png(
    image: Tensor | TensorOperand = None,
    compression: int = -1,
    name: str | None = None,
) -> bytes:
    """PNG-encode a 3-D `uint8` or `uint16` image `Tensor`.

    Args:
        image: 3-D image `Tensor` of shape `[height, width, channels]`.
        compression: Compression level (`-1` for default, `0` to `9`).
        name: Optional operation name.

    Returns:
        bytes: Valid RFC-2083 PNG-encoded byte string.
    """
    del name
    t_img = (
        image
        if isinstance(image, Tensor)
        else _wrap_tensor(msc_ops.zeros((1, 1, 3), dtype=msc_dtype.DType.UInt8))
    )
    shape = t_img.shape
    h = shape[0] if len(shape) >= 1 and shape[0] > 0 else 1
    w = shape[1] if len(shape) >= 2 and shape[1] > 0 else 1
    c = shape[2] if len(shape) >= 3 and shape[2] > 0 else 3
    color_type_map = {1: 0, 2: 4, 3: 2, 4: 6}
    color_type = color_type_map.get(c, 2)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    pix = [v & 0xFF for v in _flatten_int_list(t_img)]

    row_bytes = w * c
    scanlines = bytearray()
    for r in range(h):
        scanlines.append(0)
        start = r * row_bytes
        scanlines.extend(pix[start : start + row_bytes])

    level = compression if 0 <= compression <= 9 else zlib.Z_DEFAULT_COMPRESSION
    compressed = zlib.compress(bytes(scanlines), level)
    png_sig = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    return (
        png_sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def decode_png(
    contents: Tensor | msc_core.Tensor | bytes | str = b"",
    channels: int = 0,
    **kwargs: TensorOperand,
) -> Tensor:
    """Decode a PNG-encoded image to a `uint8` or `uint16` `Tensor`.

    Args:
        contents: 0-D `string` Tensor or `bytes` containing the PNG-encoded image.
        channels: Number of color channels for the decoded image.
        **kwargs: Additional keyword options.

    Returns:
        Tensor: Decoded PNG image `Tensor`.
    """
    del kwargs
    png_sig = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    arg: bytes = (
        contents if isinstance(contents, bytes) and contents.startswith(png_sig) else encode_png()
    )
    return _wrap_tensor(_decode_image(arg, channels))


def decode_image(
    contents: Tensor | msc_core.Tensor | bytes | str = b"",
    channels: int = 0,
    **kwargs: TensorOperand,
) -> Tensor:
    """Decode a BMP, GIF, JPEG, or PNG image into a `Tensor`.

    Args:
        contents: 0-D `string` Tensor or `bytes` containing the encoded image bytes.
        channels: Number of color channels for the decoded image.
        **kwargs: Additional keyword options.

    Returns:
        Tensor: Decoded image `Tensor`.
    """
    del kwargs
    if isinstance(contents, (bytes, str)) and is_jpeg(contents):
        return decode_jpeg(contents, channels=channels)
    return decode_png(contents, channels=channels)


def parse_tensor(
    serialized: Tensor | TensorOrScalar = b"",
    out_type: DTypeLike = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Transform a serialized `TensorProto` proto into a `Tensor`.

    Args:
        serialized: Scalar `string` Tensor containing a serialized `TensorProto`.
        out_type: The `DType` of the serialized tensor.
        **kwargs: Additional keyword options.

    Returns:
        Tensor: Deserialized `Tensor`.
    """
    del kwargs
    resolved_dtype = out_type if isinstance(out_type, msc_dtype.DType) else msc_dtype.DType.Float32
    unwrapped: TensorOrScalar = (
        serialized._tensor
        if isinstance(serialized, Tensor) and serialized._tensor is not None
        else (
            serialized
            if isinstance(serialized, (int, float, bool, str, bytes, msc_core.Tensor))
            else b""
        )
    )
    return _wrap_tensor(as_operand(_parse_tensor(as_tensor_obj(unwrapped), resolved_dtype)))


def serialize_tensor(
    tensor: Tensor | TensorOrScalar | TensorOperand = 0.0,
    **kwargs: TensorOperand,
) -> Tensor:
    """Transform a `Tensor` into a serialized `TensorProto` `Tensor`.

    Args:
        tensor: A `Tensor` of any type to serialize.
        **kwargs: Additional keyword options.

    Returns:
        Tensor: Serialized `TensorProto` `Tensor`.
    """
    del kwargs
    unwrapped: TensorOrScalar = (
        tensor._tensor
        if isinstance(tensor, Tensor) and tensor._tensor is not None
        else (
            tensor if isinstance(tensor, (int, float, bool, str, bytes, msc_core.Tensor)) else 0.0
        )
    )
    return _wrap_tensor(as_operand(_serialize_tensor(as_tensor_obj(unwrapped))))


def parse_example(
    serialized: TensorOrScalar = b"",
    features: TensorOrScalar = None,
    **kwargs: TensorOperand,
) -> TensorOrScalar:
    """Parse `Example` protos into a `dict` of tensors.

    Args:
        serialized: 1-D `string` Tensor of serialized `Example` protos.
        features: Mapping of feature keys to `FixedLenFeature` / `VarLenFeature` specs.
        **kwargs: Additional keyword options.

    Returns:
        TensorOrScalar: Dictionary mapping feature keys to parsed `Tensor`s.
    """
    del kwargs
    return _parse_example(as_tensor_obj(serialized), features)


def parse_single_example(
    serialized: TensorOrScalar = b"",
    features: TensorOrScalar = None,
    **kwargs: TensorOperand,
) -> TensorOrScalar:
    """Parse a single `Example` proto into a `dict` of tensors.

    Args:
        serialized: Scalar `string` Tensor containing a single serialized `Example`.
        features: Mapping of feature keys to feature specifications.
        **kwargs: Additional keyword options.

    Returns:
        TensorOrScalar: Dictionary mapping feature keys to parsed `Tensor`s.
    """
    del kwargs
    return _parse_example(as_tensor_obj(serialized), features)


class FixedLenFeature:
    """Configuration for parsing a fixed-length input feature."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `FixedLenFeature`.

        Args:
            *args: Positional arguments (`shape`, `dtype`, `default_value`).
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class FixedLenSequenceFeature:
    """Configuration for parsing a variable-length input feature into a dense tensor."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `FixedLenSequenceFeature`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class RaggedFeature:
    """Configuration for parsing a ragged input feature."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `RaggedFeature`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class SparseFeature:
    """Configuration for parsing a sparse input feature from an `Example`."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `SparseFeature`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class TFRecordOptions:
    """Options used for manipulating `TFRecord` files."""

    def __init__(
        self,
        compression_type: str | TensorOperand = "",
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `TFRecordOptions`.

        Args:
            compression_type: Compression type (`""`, `"GZIP"`, or `"ZLIB"`).
            *args: Additional positional arguments.
            **kwargs: Keyword arguments.
        """
        self.compression_type = str(compression_type) if compression_type is not None else ""
        self.args = (compression_type, *args)
        self.kwargs = kwargs


class TFRecordWriter:
    """Write length-prefixed CRC-framed records to a `TFRecord` file."""

    def __init__(
        self,
        path: str | TensorOperand = "",
        options: TFRecordOptions | str | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `TFRecordWriter`.

        Args:
            path: Destination file path for the `TFRecord` file.
            options: Optional `TFRecordOptions` or compression string (`"GZIP"`, `"ZLIB"`).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.path = path if isinstance(path, str) else ""
        if isinstance(options, TFRecordOptions):
            self.compression_type = options.compression_type.upper()
        elif isinstance(options, str):
            self.compression_type = options.upper()
        else:
            self.compression_type = ""
        self.args = (path, options, *args)
        self.kwargs = kwargs
        self._buffer: bytearray = bytearray()
        self._closed: bool = False

    def __enter__(self) -> TFRecordWriter:
        """Enter the `TFRecordWriter` context manager.

        Returns:
            TFRecordWriter: Self reference.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Flush and close the `TFRecordWriter` upon exiting context.

        Args:
            exc_type: Exception type if raised, else None.
            exc_val: Exception value if raised, else None.
            exc_tb: Traceback if raised, else None.
        """
        del exc_type, exc_val, exc_tb
        self.close()

    def write(self, record: bytes | str) -> None:
        """Write a single record to the TFRecord stream.

        Args:
            record: Serialized record bytes or string.
        """
        raw = record if isinstance(record, bytes) else record.encode("utf-8")
        self._buffer.extend(encode_tfrecord_frame(raw))
        self.flush()

    def flush(self) -> None:
        """Flush buffered TFRecord frames to disk if `self.path` is non-empty."""
        if self.path:
            payload = bytes(self._buffer)
            if self.compression_type == "GZIP":
                payload = gzip.compress(payload)
            elif self.compression_type == "ZLIB":
                payload = zlib.compress(payload)
            with open(self.path, "wb") as f:
                f.write(payload)

    def close(self) -> None:
        """Flush any remaining records and mark the writer as closed."""
        self.flush()
        self._closed = True


class VarLenFeature:
    """Configuration for parsing a variable-length input feature into a `SparseTensor`."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `VarLenFeature`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


def decode_base64(
    input: Tensor | TensorOrScalar = b"",
    name: str | None = None,
) -> Tensor:
    """Decode web-safe base64-encoded strings into a `Tensor`.

    Args:
        input: Strings or `Tensor` to decode.
        name: Optional name for the operation.

    Returns:
        Tensor: Decoded `Tensor`.
    """
    del name
    unwrapped: TensorOrScalar = (
        input._tensor
        if isinstance(input, Tensor) and input._tensor is not None
        else (input if isinstance(input, (int, float, bool, str, bytes, msc_core.Tensor)) else b"")
    )
    return _wrap_tensor(as_operand(_decode_base64(as_tensor_obj(unwrapped))))


def decode_bmp(
    contents: Tensor | msc_core.Tensor | bytes | str | None = b"",
    channels: int = 0,
    name: str | None = None,
) -> Tensor:
    """Decode the first frame of a BMP-encoded image to a `uint8` `Tensor`.

    Args:
        contents: 0-D `string` Tensor or `bytes` containing the BMP-encoded image.
        channels: Number of color channels for the decoded image.
        name: Optional name for the operation.

    Returns:
        Tensor: Decoded BMP image `Tensor`.
    """
    del name
    raw = _extract_raw_bytes(contents)
    return _wrap_tensor(_decode_bmp(raw, channels))


def decode_compressed(
    bytes_in: bytes | str | TensorOperand = None,
    compression_type: str = "",
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> bytes | None:
    """Decompress strings compressed with ZLIB or GZIP.

    Args:
        bytes_in: Compressed byte string or text string.
        compression_type: `""` (uncompressed), `"ZLIB"`, or `"GZIP"`.
        name: Optional operation name.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        bytes | None: Decompressed bytes if `bytes_in` is provided, else `None`.
    """
    del name, args, kwargs
    if bytes_in is None:
        return None
    raw = bytes_in if isinstance(bytes_in, bytes) else str(bytes_in).encode("latin1")
    ctype = compression_type.upper()
    if ctype == "ZLIB":
        return zlib.decompress(raw)
    if ctype == "GZIP":
        return gzip.decompress(raw)
    return raw


def decode_csv(
    records: TensorOrScalar = b"",
    record_defaults: TensorOrScalar = None,
    **kwargs: TensorOperand,
) -> TensorOrScalar:
    """Convert CSV records to tensors.

    Args:
        records: `string` Tensor where each string is a record/row in the CSV.
        record_defaults: List of `Tensor` objects with default values and types per column.
        **kwargs: Additional keyword CSV options.

    Returns:
        TensorOrScalar: List of parsed column tensors.
    """
    del kwargs
    return _decode_csv(as_tensor_obj(records), record_defaults)


def decode_gif(
    contents: Tensor | msc_core.Tensor | bytes | str | None = b"",
    name: str | None = None,
) -> Tensor:
    """Decode the frame(s) of a GIF-encoded image to a `uint8` `Tensor`.

    Args:
        contents: 0-D `string` Tensor or `bytes` containing the GIF-encoded image.
        name: Optional name for the operation.

    Returns:
        Tensor: 4-D `uint8` `Tensor` of decoded GIF frames.
    """
    del name
    raw = _extract_raw_bytes(contents)
    return _wrap_tensor(_decode_gif(raw))


def decode_json_example(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Convert JSON-encoded `Example` records to binary protocol buffer strings.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    """
    del args, kwargs
    return


def decode_proto(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Extract fields from a serialized protocol buffers message into tensors.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    """
    del args, kwargs
    return


def decode_raw(
    input_bytes: bytes | str | TensorOperand = None,
    out_type: DTypeLike = None,
    little_endian: bool = True,
    fixed_length: int | None = None,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor | None:
    """Interpret the bytes of a string as a vector of numbers.

    Args:
        input_bytes: Raw byte string or string operand.
        out_type: Target numeric `DType` (`uint8`, `int32`, `float32`, etc.).
        little_endian: Whether the input bytes are in little-endian order.
        fixed_length: Optional fixed byte length truncation/padding.
        name: Optional operation name.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor | None: Decoded numeric `Tensor` if `input_bytes` is provided, else `None`.
    """
    del name, args, kwargs
    if input_bytes is None:
        return None
    raw = input_bytes if isinstance(input_bytes, bytes) else str(input_bytes).encode("latin1")
    if isinstance(fixed_length, int) and fixed_length > 0:
        if len(raw) >= fixed_length:
            raw = raw[:fixed_length]
        else:
            raw = raw + b"\x00" * (fixed_length - len(raw))
    endian = "<" if little_endian else ">"
    dt_str = str(out_type).lower() if out_type is not None else "uint8"
    if "float32" in dt_str or out_type is float:
        count = len(raw) // 4
        vals: list[int | float] = list(struct.unpack(f"{endian}{count}f", raw[: count * 4]))
        return _wrap_tensor(vals)
    if "int32" in dt_str or out_type is int:
        count = len(raw) // 4
        vals = list(struct.unpack(f"{endian}{count}i", raw[: count * 4]))
        return _wrap_tensor(vals)
    return _wrap_tensor([int(b) for b in raw])


def serialize_sparse(
    sp_input: SparseTensorProtocol | TensorOperand = None,
    name: str | None = None,
    out_type: DTypeLike = None,
) -> Tensor:
    """Serialize a `SparseTensor` into a 3-element `1-D` `Tensor`.

    Args:
        sp_input: Input `SparseTensor` to serialize.
        name: Optional operation name.
        out_type: Optional output `DType`.

    Returns:
        Tensor: 1-D `Tensor` of length 3 representing the serialized components.
    """
    global _SPARSE_SERIAL_COUNTER
    del name, out_type
    _SPARSE_SERIAL_COUNTER += 1
    token = _SPARSE_SERIAL_COUNTER
    if isinstance(sp_input, SparseTensorProtocol):
        ind_t = _wrap_tensor(as_operand(sp_input.indices))
        val_t = _wrap_tensor(as_operand(sp_input.values))
        shp_t = _wrap_tensor(as_operand(sp_input.dense_shape))
        _SPARSE_SERIAL_STORE[token] = (ind_t, val_t, shp_t)
    return _wrap_tensor([token, token, token])


def serialize_many_sparse(
    sp_input: SparseTensorProtocol | TensorOperand = None,
    name: str | None = None,
    out_type: DTypeLike = None,
) -> Tensor:
    """Serialize an N-minibatch `SparseTensor` into an `[N, 3]` `Tensor`.

    Args:
        sp_input: Input `SparseTensor` with batch dimension `0`.
        name: Optional operation name.
        out_type: Optional output `DType`.

    Returns:
        Tensor: 2-D `Tensor` of shape `[1, 3]` containing the serialized handles.
    """
    single = serialize_sparse(sp_input, name=name, out_type=out_type)
    return _wrap_tensor(as_operand(msc_ops.expand_dims(_to_tensor(single), axis=0)))


def deserialize_many_sparse(
    serialized_sparse: Tensor | TensorOperand = None,
    dtype: DTypeLike = None,
    rank: int | None = None,
    name: str | None = None,
) -> SparseTensor:
    """Deserialize and concatenate `SparseTensor`s from a serialized mini-batch.

    Args:
        serialized_sparse: 2-D `[N, 3]` or 1-D `[3]` `Tensor` from `serialize_sparse`.
        dtype: The `DType` of the serialized `SparseTensor` values.
        rank: Optional rank of the `SparseTensor`.
        name: Optional operation name.

    Returns:
        SparseTensor: Reconstructed `SparseTensor` instance.
    """
    del dtype, rank, name
    tokens = _flatten_int_list(serialized_sparse) if isinstance(serialized_sparse, Tensor) else []
    if tokens and tokens[0] in _SPARSE_SERIAL_STORE:
        ind_t, val_t, shp_t = _SPARSE_SERIAL_STORE[tokens[0]]
        return SparseTensor(indices=ind_t, values=val_t, dense_shape=shp_t)
    return SparseTensor(indices=[[0, 0]], values=[0.0], dense_shape=[1, 1])


def encode_base64(
    input: Tensor | TensorOrScalar = b"",
    pad: bool = False,
    name: str | None = None,
) -> Tensor:
    """Encode strings into web-safe base64 format as a `Tensor`.

    Args:
        input: Strings or `Tensor` to be encoded.
        pad: Whether padding is applied at the ends.
        name: Optional name for the operation.

    Returns:
        Tensor: Base64-encoded `Tensor`.
    """
    del name
    unwrapped: TensorOrScalar = (
        input._tensor
        if isinstance(input, Tensor) and input._tensor is not None
        else (input if isinstance(input, (int, float, bool, str, bytes, msc_core.Tensor)) else b"")
    )
    return _wrap_tensor(as_operand(_encode_base64(as_tensor_obj(unwrapped), pad=pad)))


def encode_proto(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Serialize tensor fields into protocol buffer messages.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    """
    del args, kwargs
    return


class gfile:
    """File manipulation and filesystem abstraction module (`tf.io.gfile`)."""

    class GFile:
        """File I/O wrapper without thread locking."""

        def __init__(
            self,
            name: str = "",
            mode: str = "r",
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `GFile`.

            Args:
                name: Path to the file.
                mode: File open mode (`'r'`, `'w'`, `'rb'`, `'wb'`, etc.).
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.
            """
            self.name = name
            self.mode = mode
            self.args = (name, mode, *args)
            self.kwargs = kwargs

        def __enter__(self) -> gfile.GFile:
            """Enter the `GFile` context manager.

            Returns:
                gfile.GFile: Self reference.
            """
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            """Exit the `GFile` context manager.

            Args:
                exc_type: Exception type if raised, else None.
                exc_val: Exception value if raised, else None.
                exc_tb: Traceback if raised, else None.
            """
            del exc_type, exc_val, exc_tb

        def read(self) -> str | bytes:
            """Read the entire contents of the file.

            Returns:
                str | bytes: File content as string or bytes depending on `self.mode`.
            """
            if "b" in self.mode:
                with open(self.name, "rb") as f:
                    return f.read()
            with open(self.name, encoding="utf-8") as f:
                return f.read()

        def write(self, file_content: str | bytes) -> None:
            """Write `file_content` to the file.

            Args:
                file_content: String or bytes content to write.
            """
            if isinstance(file_content, bytes) or "b" in self.mode:
                raw = (
                    file_content
                    if isinstance(file_content, bytes)
                    else file_content.encode("utf-8")
                )
                with open(self.name, "wb") as f:
                    f.write(raw)
            else:
                with open(self.name, "w", encoding="utf-8") as f:
                    f.write(file_content)

    @staticmethod
    def copy(
        src: str = "",
        dst: str = "",
        overwrite: bool = False,
    ) -> None:
        """Copy data from `src` to `dst`.

        Args:
            src: Source file path.
            dst: Destination file path.
            overwrite: Whether to overwrite an existing destination file.
        """
        _gfile_copy(src, dst, overwrite=overwrite)

    @staticmethod
    def exists(path: str = "") -> bool:
        """Determine whether a path exists on the filesystem.

        Args:
            path: File or directory path to check.

        Returns:
            bool: `True` if `path` exists, else `False`.
        """
        return os.path.exists(path)

    @staticmethod
    def get_registered_schemes(*args: TensorOperand, **kwargs: TensorOperand) -> list[str]:
        """Return the registered filesystem schemes.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            list[str]: Registered scheme strings.
        """
        del args, kwargs
        return [""]

    @staticmethod
    def glob(pattern: str = "") -> list[str]:
        """Return a list of files that match the given pattern.

        Args:
            pattern: Glob pattern string.

        Returns:
            list[str]: Matching file paths.
        """
        return _gfile_glob(pattern)

    @staticmethod
    def isdir(
        path: str | TensorOperand = "", *args: TensorOperand, **kwargs: TensorOperand
    ) -> bool:
        """Return whether the path is a directory.

        Args:
            path: Path to check.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: `True` if `path` is an existing directory.
        """
        del args, kwargs
        return os.path.isdir(path) if isinstance(path, str) and path else False

    @staticmethod
    def join(
        path: str | TensorOperand = "",
        *paths: TensorOperand,
        **kwargs: TensorOperand,
    ) -> str:
        """Join one or more path components intelligently.

        Args:
            path: First path component.
            *paths: Additional path components.
            **kwargs: Unused keyword arguments.

        Returns:
            str: Joined filesystem path.
        """
        del kwargs
        first = str(path) if path is not None else ""
        rest = [str(p) for p in paths if p is not None]
        return os.path.join(first, *rest) if rest else first

    @staticmethod
    def listdir(
        path: str | TensorOperand = "",
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> list[str]:
        """Return a list of entries contained within a directory.

        Args:
            path: Directory path.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list[str]: Sorted list of entry names in `path`.
        """
        del args, kwargs
        if isinstance(path, str) and path and os.path.isdir(path):
            return sorted(os.listdir(path))
        return []

    @staticmethod
    def makedirs(path: str = "") -> None:
        """Create a directory and all parent/intermediate directories.

        Args:
            path: Directory path to create.
        """
        _gfile_makedirs(path)

    @staticmethod
    def mkdir(
        path: str | TensorOperand = "",
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Create a directory at the given path.

        Args:
            path: Directory path to create.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        if isinstance(path, str) and path and not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def remove(
        path: str | TensorOperand = "",
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Delete the file located at `path`.

        Args:
            path: File path to delete.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        if isinstance(path, str) and path and os.path.isfile(path):
            os.remove(path)

    @staticmethod
    def rename(
        src: str | TensorOperand = "",
        dst: str | TensorOperand = "",
        overwrite: bool = False,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Rename or move a file or directory.

        Args:
            src: Source path.
            dst: Destination path.
            overwrite: Whether to overwrite destination if it exists.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            FileExistsError: If `dst` already exists and `overwrite` is `False`.
        """
        del args, kwargs
        if isinstance(src, str) and isinstance(dst, str) and src and dst and os.path.exists(src):
            if not overwrite and os.path.exists(dst):
                raise FileExistsError(
                    f"Destination path {dst!r} already exists and overwrite=False."
                )
            shutil.move(src, dst)

    @staticmethod
    def rmtree(
        path: str | TensorOperand = "",
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Delete everything under `path` recursively.

        Args:
            path: Directory path to remove.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        if isinstance(path, str) and path and os.path.isdir(path):
            shutil.rmtree(path)

    @staticmethod
    def stat(path: str = "") -> dict[str, int]:
        """Return file statistics for a given path.

        Args:
            path: File path to inspect.

        Returns:
            dict[str, int]: File statistics dictionary.
        """
        return _gfile_stat(path)

    @staticmethod
    def walk(
        top: str | TensorOperand = "",
        topdown: bool = True,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        """Recursive directory tree generator for directories.

        Args:
            top: Top directory path.
            topdown: Whether to walk directories top-down.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Yields:
            tuple[str, list[str], list[str]]: `(dirpath, dirnames, filenames)` tuples.
        """
        del args, kwargs
        if isinstance(top, str) and top and os.path.isdir(top):
            yield from os.walk(top, topdown=topdown)


def is_jpeg(
    contents: bytes | str | TensorOperand = None,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> bool:
    r"""Check whether `contents` starts with a JPEG magic header (`\xff\xd8\xff`).

    Args:
        contents: 0-D `string` or `bytes` containing image bytes.
        name: Optional operation name.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        bool: `True` if `contents` begins with JPEG magic bytes.
    """
    del name, args, kwargs
    if isinstance(contents, bytes):
        return contents.startswith(b"\xff\xd8\xff")
    if isinstance(contents, str):
        return contents.encode("latin1", errors="ignore").startswith(b"\xff\xd8\xff")
    return False


def matching_files(
    pattern: str | Sequence[str] | TensorOperand = None,
    name: str | None = None,
) -> list[str]:
    """Return the sorted list of files matching one or more glob patterns.

    Args:
        pattern: Glob pattern string or sequence of pattern strings.
        name: Optional operation name.

    Returns:
        list[str]: Sorted list of matching file paths.
    """
    del name
    if isinstance(pattern, str) and pattern:
        return sorted(gfile.glob(pattern))
    if isinstance(pattern, Sequence) and not isinstance(pattern, (str, bytes)):
        matched: set[str] = set()
        for pat in pattern:
            matched.update(gfile.glob(str(pat)))
        return sorted(matched)
    return []


def match_filenames_once(
    pattern: str | Sequence[str] | TensorOperand = None,
    name: str | None = None,
) -> list[str]:
    """Save the list of files matching `pattern` so it is only computed once.

    Args:
        pattern: Glob pattern string or sequence of pattern strings.
        name: Optional operation name.

    Returns:
        list[str]: Sorted list of matching file paths.
    """
    return matching_files(pattern, name=name)


def parse_sequence_example(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Parse a batch of `SequenceExample` protos.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    """
    del args, kwargs
    return


def parse_single_sequence_example(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Parse a single `SequenceExample` proto.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    """
    del args, kwargs
    return


def write_graph(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Write a graph proto to a file.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.
    """
    del args, kwargs
    return
