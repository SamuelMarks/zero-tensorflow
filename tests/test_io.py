"""Unit tests for `test_io`."""

from __future__ import annotations

import gzip
import struct
import zlib
from pathlib import Path

from zero_tensorflow import Tensor, debugging, io, sparse

import pytest


def test_read_write_file(tmp_path: Path) -> None:
    """Verify `io.write_file`, `io.read_file`, `io.matching_files`, `io.match_filenames_once`, and `gfile.rename`.

    Args:
        tmp_path: Temporary directory fixture.
    """
    p = tmp_path / "test.txt"
    io.write_file(str(p), b"hello world")
    res = io.read_file(str(p))
    assert res == b"hello world"

    p2 = tmp_path / "other.txt"
    io.write_file(str(p2), Tensor(1.0))

    matched = io.matching_files(str(tmp_path / "*.txt"))
    assert len(matched) == 2
    matched_multi = io.match_filenames_once([str(tmp_path / "*.txt")])
    assert matched_multi == matched
    assert io.matching_files(None) == []

    with pytest.raises(FileExistsError, match="already exists"):
        io.gfile.rename(str(p), str(p2), overwrite=False)
    io.gfile.rename(str(p), str(p2), overwrite=True)
    assert not p.exists() and p2.exists()


def test_io_serialization_and_compression(tmp_path: Path) -> None:
    """Verify `io` JPEG/PNG encoding/decoding, sparse serialization, TFRecord framing, and `decode_raw`.

    Args:
        tmp_path: Temporary directory fixture.
    """
    img = Tensor([[[10, 20, 30], [40, 50, 60]]], dtype="uint8")
    jpeg_bytes = io.encode_jpeg(img)
    assert io.is_jpeg(jpeg_bytes) is True
    j_shape = io.extract_jpeg_shape(jpeg_bytes)
    debugging.assert_equal(j_shape, [1, 2, 3])
    decoded_j = io.decode_jpeg(jpeg_bytes)
    assert isinstance(decoded_j, Tensor)
    debugging.assert_equal(decoded_j, img)
    assert io.decode_image(jpeg_bytes).shape == (1, 2, 3)

    cropped_j = io.decode_and_crop_jpeg(jpeg_bytes, crop_window=[0, 1, 1, 1])
    assert cropped_j.shape == (1, 1, 3)

    png_bytes = io.encode_png(img, compression=6)
    decoded_p = io.decode_png(png_bytes)
    assert isinstance(decoded_p, Tensor)
    assert decoded_p.shape == (1, 2, 3)
    assert isinstance(io.decode_png(Tensor(1.0)), Tensor)
    assert isinstance(io.decode_image(Tensor(1.0)), Tensor)
    assert isinstance(io.decode_bmp(Tensor(1.0)), Tensor)
    assert isinstance(io.decode_gif(Tensor(1.0)), Tensor)
    assert isinstance(io.decode_base64(Tensor(1.0)), Tensor)
    assert isinstance(io.encode_base64(Tensor(1.0)), Tensor)

    serialized = io.serialize_tensor(Tensor([1.0, 2.0]))
    assert isinstance(serialized, Tensor)
    parsed = io.parse_tensor(serialized)
    assert isinstance(parsed, Tensor)

    sp = sparse.SparseTensor(indices=[[0, 1]], values=[5.0], dense_shape=[2, 2])
    ser_sp = io.serialize_sparse(sp)
    assert ser_sp.shape == (3,)
    ser_many = io.serialize_many_sparse(sp)
    assert ser_many.shape == (1, 3)
    deser_sp = io.deserialize_many_sparse(ser_many)
    assert isinstance(deser_sp.values, Tensor)
    debugging.assert_near(deser_sp.values, [5.0])

    # TFRecordWriter and decode_tfrecord_frames round-trip
    io.TFRecordWriter().close()
    tfr_path = str(tmp_path / "data.tfrecord")
    with io.TFRecordWriter(tfr_path) as writer:
        writer.write(b"rec1")
        writer.write("rec2")
    raw_tfr = io.read_file(tfr_path)
    assert io.decode_tfrecord_frames(raw_tfr) == [b"rec1", b"rec2"]
    with pytest.raises(ValueError, match="Truncated TFRecord frame"):
        io.decode_tfrecord_frames(b"short")
    with pytest.raises(ValueError, match="Corrupted TFRecord CRC"):
        io.decode_tfrecord_frames(raw_tfr[:-1] + bytes([raw_tfr[-1] ^ 0xFF]))

    tfr_gz_path = str(tmp_path / "data.tfrecord.gz")
    with io.TFRecordWriter(tfr_gz_path, options=io.TFRecordOptions("GZIP")) as gz_writer:
        gz_writer.write(b"gz_rec")
    raw_gz = io.decode_compressed(io.read_file(tfr_gz_path), compression_type="GZIP")
    assert raw_gz is not None and io.decode_tfrecord_frames(raw_gz) == [b"gz_rec"]

    tfr_zlib_path = str(tmp_path / "data.tfrecord.zlib")
    with io.TFRecordWriter(tfr_zlib_path, options="ZLIB") as zlib_writer:
        zlib_writer.write(b"zlib_rec")
    raw_zl = io.decode_compressed(io.read_file(tfr_zlib_path), compression_type="ZLIB")
    assert raw_zl is not None and io.decode_tfrecord_frames(raw_zl) == [b"zlib_rec"]

    zlib_bytes = zlib.compress(b"compressed_payload")
    assert io.decode_compressed(zlib_bytes, compression_type="ZLIB") == b"compressed_payload"

    gzip_bytes = gzip.compress(b"gzip_payload")
    assert io.decode_compressed(gzip_bytes, compression_type="GZIP") == b"gzip_payload"

    f32_bytes = struct.pack("<2f", 1.5, -2.5)
    decoded_f32 = io.decode_raw(f32_bytes, out_type=float, little_endian=True)
    assert isinstance(decoded_f32, Tensor)
    debugging.assert_near(decoded_f32, [1.5, -2.5])

    i32_bytes = struct.pack(">2i", 10, 20)
    decoded_i32 = io.decode_raw(i32_bytes, out_type=int, little_endian=False)
    assert isinstance(decoded_i32, Tensor)
    debugging.assert_equal(decoded_i32, [10, 20])

    trunc = io.decode_raw(b"\x01\x02\x03\x04", out_type="uint8", fixed_length=2)
    assert isinstance(trunc, Tensor)
    debugging.assert_equal(trunc, [1, 2])
    padded = io.decode_raw(b"\x01\x02", out_type="uint8", fixed_length=4)
    assert isinstance(padded, Tensor)
    debugging.assert_equal(padded, [1, 2, 0, 0])
