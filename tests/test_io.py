"""Unit tests for `test_io`."""

from __future__ import annotations

import gzip
import struct
import zlib
from pathlib import Path

from zero_tensorflow import Tensor, io

import numpy as np


def test_read_write_file(tmp_path: Path) -> None:
    """Verify `io.write_file` and `io.read_file`.

    Args:
        tmp_path: Temporary directory fixture.
    """
    p = tmp_path / "test.txt"
    io.write_file(str(p), b"hello world")
    res = io.read_file(str(p))
    assert res == b"hello world"


def test_io_serialization_and_compression() -> None:
    """Verify `io` serialization, parsing, compression, and raw byte decoding."""
    io.decode_jpeg(b"dummy")

    serialized = io.serialize_tensor(Tensor([1.0, 2.0]))
    assert serialized is not None
    parsed = io.parse_tensor(serialized)
    assert parsed is not None

    zlib_bytes = zlib.compress(b"compressed_payload")
    assert io.decode_compressed(zlib_bytes, compression_type="ZLIB") == b"compressed_payload"

    gzip_bytes = gzip.compress(b"gzip_payload")
    assert io.decode_compressed(gzip_bytes, compression_type="GZIP") == b"gzip_payload"

    f32_bytes = struct.pack("<2f", 1.5, -2.5)
    decoded_f32 = io.decode_raw(f32_bytes, out_type=float, little_endian=True)
    assert isinstance(decoded_f32, Tensor)
    np.testing.assert_allclose(decoded_f32.numpy(), [1.5, -2.5])

    i32_bytes = struct.pack(">2i", 10, 20)
    decoded_i32 = io.decode_raw(i32_bytes, out_type=int, little_endian=False)
    assert isinstance(decoded_i32, Tensor)
    np.testing.assert_array_equal(decoded_i32.numpy(), [10, 20])
