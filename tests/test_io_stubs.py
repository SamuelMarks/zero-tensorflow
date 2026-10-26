"""Behavioral unit tests for `zero_tensorflow.io` feature specs, codecs, and file operations."""

from __future__ import annotations

from pathlib import Path

import zero_tensorflow as tf
import zero_tensorflow.io as tf_io
from zero_tensorflow.sparse import SparseTensor


def test_io_stubs(tmp_path: Path) -> None:
    """Verify `tf.io` feature specifications, image/base64 codecs, sparse serialization, and file I/O.

    Args:
        tmp_path: Temporary directory fixture.
    """
    fixed = tf_io.FixedLenFeature(shape=[2], dtype="int64", default_value=[0, 0])
    assert fixed.kwargs["shape"] == [2]
    assert fixed.kwargs["dtype"] == "int64"
    assert fixed.kwargs["default_value"] == [0, 0]

    seq_feat = tf_io.FixedLenSequenceFeature(shape=[3], dtype="float32", allow_missing=True)
    assert seq_feat.kwargs["shape"] == [3]
    assert seq_feat.kwargs["allow_missing"] is True

    ragged_feat = tf_io.RaggedFeature(dtype="int64", value_key="vals")
    assert ragged_feat.kwargs["dtype"] == "int64"
    assert ragged_feat.kwargs["value_key"] == "vals"

    sparse_feat = tf_io.SparseFeature(
        index_key="idx",
        value_key="val",
        dtype="float32",
        size=10,
    )
    assert sparse_feat.kwargs["index_key"] == "idx"
    assert sparse_feat.kwargs["value_key"] == "val"
    assert sparse_feat.kwargs["size"] == 10

    var_feat = tf_io.VarLenFeature(dtype="string")
    assert var_feat.kwargs["dtype"] == "string"

    opts = tf_io.TFRecordOptions(compression_type="GZIP")
    assert opts.compression_type == "GZIP"

    tfr_file = tmp_path / "sample.tfrecord"
    with tf_io.TFRecordWriter(str(tfr_file)) as writer:
        writer.write(b"record_alpha")
        writer.flush()
    assert tfr_file.exists()

    b64 = tf_io.encode_base64("hello!")
    assert isinstance(b64, tf.Tensor)
    decoded_b64 = tf_io.decode_base64(b64)
    assert isinstance(decoded_b64, tf.Tensor)

    assert tf_io.parse_example('{"x": [1.0]}', {}) == {}
    assert tf_io.parse_single_example('{"x": [1.0]}', {}) == {}
    tf_io.parse_sequence_example()
    tf_io.parse_single_sequence_example()

    png_bytes = tf_io.encode_png(tf.zeros((2, 2, 3), dtype=tf.uint8))
    assert isinstance(png_bytes, bytes)
    dec_png = tf_io.decode_png(png_bytes)
    assert isinstance(dec_png, tf.Tensor)
    assert dec_png.shape == (2, 2, 3)

    jpeg_bytes = tf_io.encode_jpeg(tf.zeros((4, 3, 3), dtype=tf.uint8))
    assert isinstance(jpeg_bytes, bytes)
    assert tf_io.is_jpeg(jpeg_bytes) is True
    jpeg_shape = tf_io.extract_jpeg_shape(jpeg_bytes)
    assert jpeg_shape is not None
    tf.debugging.assert_equal(jpeg_shape, [4, 3, 3])

    dec_jpeg = tf_io.decode_jpeg(jpeg_bytes)
    assert dec_jpeg.shape == (4, 3, 3)
    crop_jpeg = tf_io.decode_and_crop_jpeg(jpeg_bytes, crop_window=[1, 1, 2, 2])
    assert crop_jpeg.shape == (2, 2, 3)

    assert isinstance(tf_io.decode_bmp(b""), tf.Tensor)
    bmp_header = (
        b"BM"
        + b"\x3a\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00"
        + b"\x28\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00"
        + b"\x00" * 24
        + b"\x1e\x14\x0a\x00"
    )
    tf.debugging.assert_equal(tf_io.decode_bmp(bmp_header), [[[10, 20, 30]]])
    tf.debugging.assert_equal(tf_io.decode_image(bmp_header), [[[10, 20, 30]]])
    gif_header = b"GIF89a\x02\x00\x03\x00\x00\x00\x00"
    assert tf_io.decode_gif(gif_header).shape == (1, 3, 2, 3)
    assert tf_io.decode_image(gif_header).shape == (1, 3, 2, 3)
    assert isinstance(tf_io.decode_gif(b""), tf.Tensor)
    assert isinstance(tf_io.decode_image(b""), tf.Tensor)

    raw_dec = tf_io.decode_raw(b"\x01\x02\x03\x04", out_type=tf.uint8, fixed_length=6)
    assert raw_dec is not None
    tf.debugging.assert_equal(raw_dec, [1, 2, 3, 4, 0, 0])

    ser_t = tf_io.serialize_tensor(tf.constant([1.5, 2.5]))
    assert isinstance(ser_t, tf.Tensor)
    parsed_t = tf_io.parse_tensor(ser_t, out_type=tf.float32)
    assert isinstance(parsed_t, tf.Tensor)

    sp = SparseTensor(indices=[[0, 1], [1, 0]], values=[5.0, 6.0], dense_shape=[2, 2])
    ser_sp = tf_io.serialize_sparse(sp)
    assert isinstance(ser_sp, tf.Tensor)
    ser_many = tf_io.serialize_many_sparse(sp)
    assert isinstance(ser_many, tf.Tensor)
    deser_sp = tf_io.deserialize_many_sparse(ser_sp, dtype=tf.float32)
    assert isinstance(deser_sp, SparseTensor)

    txt_file = tmp_path / "note.txt"
    tf_io.write_file(str(txt_file), "content_123")
    read_back = tf_io.read_file(str(txt_file))
    assert b"content_123" in read_back

    matched = tf_io.matching_files(str(tmp_path / "*.txt"))
    assert matched is not None
    matched_once = tf_io.match_filenames_once(str(tmp_path / "*.txt"))
    assert matched_once is not None

    written_txt = tf_io.write_graph({"nodes": []}, str(tmp_path), "graph.pbtxt", as_text=True)
    assert written_txt is not None
    assert (tmp_path / "graph.pbtxt").is_file()
    written_bin = tf_io.write_graph({"nodes": []}, str(tmp_path), "graph.pb", as_text=False)
    assert written_bin is not None
    assert (tmp_path / "graph.pb").is_file()
    assert tf_io.write_graph({"nodes": []}) is None
