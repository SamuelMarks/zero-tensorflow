import zero_tensorflow.io as tf_io


def call_safely(func):
    try:
        return func()
    except Exception:  # noqa: BLE001
        return None


def test_io_stubs():
    call_safely(tf_io.FixedLenFeature)
    call_safely(tf_io.FixedLenSequenceFeature)
    call_safely(tf_io.RaggedFeature)
    call_safely(tf_io.SparseFeature)
    call_safely(tf_io.TFRecordOptions)
    call_safely(tf_io.TFRecordWriter)
    call_safely(tf_io.VarLenFeature)
    call_safely(tf_io.decode_and_crop_jpeg)
    call_safely(tf_io.decode_base64)
    call_safely(tf_io.decode_bmp)
    call_safely(tf_io.decode_compressed)
    call_safely(tf_io.decode_csv)
    call_safely(tf_io.decode_gif)
    call_safely(tf_io.decode_image)
    call_safely(tf_io.decode_jpeg)
    call_safely(tf_io.decode_json_example)
    call_safely(tf_io.decode_png)
    call_safely(tf_io.decode_proto)
    call_safely(tf_io.decode_raw)
    call_safely(tf_io.encode_base64)
    call_safely(tf_io.encode_jpeg)
    call_safely(tf_io.encode_png)
    call_safely(tf_io.extract_jpeg_shape)
    call_safely(tf_io.is_jpeg)
    call_safely(tf_io.match_filenames_once)
    call_safely(tf_io.matching_files)
    call_safely(tf_io.parse_example)
    call_safely(tf_io.parse_sequence_example)
    call_safely(tf_io.parse_single_example)
    call_safely(tf_io.parse_single_sequence_example)
    call_safely(tf_io.parse_tensor)
    call_safely(tf_io.read_file)
    call_safely(tf_io.serialize_many_sparse)
    call_safely(tf_io.serialize_sparse)
    call_safely(tf_io.serialize_tensor)
    call_safely(tf_io.write_file)
    call_safely(tf_io.write_graph)
