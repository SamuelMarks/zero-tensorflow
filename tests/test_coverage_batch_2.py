"""Unit tests for `test_coverage_batch_2`."""

from __future__ import annotations

from pathlib import Path

import zero_tensorflow as tf
from zero_tensorflow._typing import TensorOperand

import numpy as np


def test_distribute_stubs() -> None:
    """Verify `tf.distribute` classes, scopes, `run`, `reduce`, and strategy state functions."""
    assert tf.distribute.CrossDeviceOps() is not None
    assert tf.distribute.DistributedDataset() is not None
    assert tf.distribute.DistributedIterator() is not None
    assert tf.distribute.DistributedValues() is not None
    assert tf.distribute.HierarchicalCopyAllReduce() is not None
    assert tf.distribute.InputContext() is not None
    assert tf.distribute.InputOptions() is not None
    assert tf.distribute.InputReplicationMode() is not None
    assert tf.distribute.NcclAllReduce() is not None
    assert isinstance(tf.distribute.ParameterServerStrategy(), tf.distribute.Strategy)
    assert tf.distribute.ReductionToOneDevice() is not None
    assert tf.distribute.ReplicaContext() is not None
    assert tf.distribute.RunOptions() is not None
    assert tf.distribute.Server() is not None
    assert tf.distribute.StrategyExtended() is not None
    assert tf.distribute.cluster_resolver.ClusterResolver() is not None
    assert tf.distribute.cluster_resolver.GCEClusterResolver() is not None
    assert tf.distribute.cluster_resolver.KubernetesClusterResolver() is not None
    assert tf.distribute.cluster_resolver.SimpleClusterResolver() is not None
    assert tf.distribute.cluster_resolver.SlurmClusterResolver() is not None
    assert tf.distribute.cluster_resolver.TFConfigClusterResolver() is not None
    assert tf.distribute.cluster_resolver.TPUClusterResolver() is not None
    assert tf.distribute.cluster_resolver.UnionResolver() is not None
    assert tf.distribute.coordinator.ClusterCoordinator() is not None
    assert tf.distribute.coordinator.PerWorkerValue() is not None
    assert tf.distribute.coordinator.RemoteValue() is not None
    assert tf.distribute.coordinator.experimental_get_current_worker_index() is None
    assert isinstance(tf.distribute.experimental.CentralStorageStrategy(), tf.distribute.Strategy)
    assert tf.distribute.experimental.CollectiveCommunication() is not None
    assert tf.distribute.experimental.CollectiveHints() is not None
    assert tf.distribute.experimental.CommunicationImplementation() is not None
    assert tf.distribute.experimental.CommunicationOptions() is not None
    assert isinstance(
        tf.distribute.experimental.MultiWorkerMirroredStrategy(), tf.distribute.Strategy
    )
    assert isinstance(tf.distribute.experimental.ParameterServerStrategy(), tf.distribute.Strategy)
    assert tf.distribute.experimental.PreemptionCheckpointHandler() is not None
    assert tf.distribute.experimental.PreemptionWatcher() is not None
    assert isinstance(tf.distribute.experimental.TPUStrategy(), tf.distribute.Strategy)
    assert tf.distribute.experimental.TerminationConfig() is not None
    assert tf.distribute.experimental.ValueContext() is not None
    assert tf.distribute.experimental.coordinator() is None
    assert tf.distribute.experimental.partitioners() is None
    assert tf.distribute.experimental.rpc() is None

    tf.distribute.experimental_set_strategy(None)
    assert tf.distribute.has_strategy() is False
    assert isinstance(tf.distribute.get_strategy(), tf.distribute.Strategy)
    assert tf.distribute.get_replica_context() is None
    assert tf.distribute.in_cross_replica_context() is True

    ms = tf.distribute.MirroredStrategy()
    with ms.scope():
        assert tf.distribute.has_strategy() is True
        assert tf.distribute.get_strategy() is ms

        def _plus_one(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
            """Add 1 to `x` and verify active replica context."""
            assert tf.distribute.in_cross_replica_context() is False
            assert isinstance(tf.distribute.get_replica_context(), tf.distribute.ReplicaContext)
            val = args[0] if args else kwargs.get("x", 0)
            return (int(val) if isinstance(val, int) else 0) + 1

        res = ms.run(_plus_one, args=(3,))
        assert res == 4
        res2 = ms.run(_plus_one, args=(), kwargs={"x": 3})
        assert res2 == 4
        assert ms.reduce("SUM", 5) == 5
        assert ms.reduce("SUM", [2.0, 4.0]) == 6.0
        assert ms.reduce("MEAN", [2.0, 4.0]) == 3.0

    assert tf.distribute.has_strategy() is False
    tf.distribute.experimental_set_strategy(ms)
    assert tf.distribute.has_strategy() is True
    assert tf.distribute.get_strategy() is ms
    tf.distribute.experimental_set_strategy(None)
    assert tf.distribute.has_strategy() is False

    assert tf.distribute.MirroredStrategy(devices=["/cpu:0"])._devices == ["/cpu:0"]
    assert tf.distribute.MultiWorkerMirroredStrategy() is not None
    assert tf.distribute.OneDeviceStrategy(device="/cpu:0")._device == "/cpu:0"
    assert tf.distribute.TPUStrategy() is not None


def test_io_stubs(tmp_path: Path) -> None:
    """Verify `tf.io` and `tf.io.gfile` operations with concrete file and byte assertions.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    fpath = tmp_path / "test.txt"
    tf.io.write_file(str(fpath), b"hello")
    res = tf.io.read_file(str(fpath))
    assert res == b"hello"
    tf.io.write_file(str(fpath), "world")
    assert tf.io.read_file(str(fpath)) == b"world"

    tf.io.decode_jpeg(b"")
    assert tf.io.FixedLenFeature() is not None
    assert tf.io.FixedLenSequenceFeature() is not None
    assert tf.io.RaggedFeature() is not None
    assert tf.io.SparseFeature() is not None
    assert tf.io.TFRecordOptions() is not None
    assert tf.io.TFRecordWriter() is not None
    assert tf.io.VarLenFeature() is not None
    tf.io.decode_and_crop_jpeg()
    assert tf.io.decode_compressed() is None
    assert tf.io.decode_compressed(b"raw") == b"raw"
    tf.io.decode_json_example()
    tf.io.decode_proto()
    assert tf.io.decode_raw() is None
    raw_u8 = tf.io.decode_raw(b"\x01\x02", out_type="uint8")
    assert isinstance(raw_u8, tf.Tensor)
    np.testing.assert_array_equal(raw_u8.numpy(), [1, 2])
    tf.io.deserialize_many_sparse()
    tf.io.encode_jpeg()
    tf.io.encode_png()
    tf.io.encode_proto()
    tf.io.extract_jpeg_shape()

    assert tf.io.is_jpeg(b"\xff\xd8\xff\xe0") is True
    assert tf.io.is_jpeg("\xff\xd8\xff") is True
    assert tf.io.is_jpeg(b"PNG") is False
    assert tf.io.is_jpeg(None) is False

    sub_dir = tmp_path / "gfile_dir"
    tf.io.gfile.mkdir(str(sub_dir))
    assert tf.io.gfile.isdir(str(sub_dir)) is True
    joined = tf.io.gfile.join(str(sub_dir), "item.txt")
    with tf.io.gfile.GFile(joined, "w") as gf_w:
        gf_w.write("gfile_text")
    with tf.io.gfile.GFile(joined, "r") as gf_r:
        assert gf_r.read() == "gfile_text"

    bin_path = tf.io.gfile.join(str(sub_dir), "item.bin")
    with tf.io.gfile.GFile(bin_path, "wb") as gf_wb:
        gf_wb.write(b"\x00\x01")
    with tf.io.gfile.GFile(bin_path, "rb") as gf_rb:
        assert gf_rb.read() == b"\x00\x01"

    assert tf.io.gfile.listdir(str(sub_dir)) == ["item.bin", "item.txt"]
    walked = list(tf.io.gfile.walk(str(sub_dir)))
    assert len(walked) == 1 and sorted(walked[0][2]) == ["item.bin", "item.txt"]

    renamed_path = tf.io.gfile.join(str(sub_dir), "renamed.txt")
    tf.io.gfile.rename(joined, renamed_path)
    assert tf.io.gfile.exists(renamed_path) is True
    tf.io.gfile.remove(renamed_path)
    assert tf.io.gfile.exists(renamed_path) is False
    tf.io.gfile.rmtree(str(sub_dir))
    assert tf.io.gfile.isdir(str(sub_dir)) is False
    assert tf.io.gfile.get_registered_schemes() == [""]

    tf.io.match_filenames_once()
    tf.io.matching_files()
    tf.io.parse_sequence_example()
    tf.io.parse_single_sequence_example()
    tf.io.serialize_many_sparse()
    tf.io.serialize_sparse()
    tf.io.write_graph()


def test_saved_model_stubs(tmp_path: Path) -> None:
    """Verify `tf.saved_model` classes and functions.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    sm_dir = str(tmp_path / "sm")
    tf.saved_model.save(lambda *args, **kwargs: None, sm_dir)
    tf.saved_model.load(sm_dir)
    assert tf.saved_model.Asset("vocab.txt") is not None
    assert tf.saved_model.LoadOptions() is not None
    assert tf.saved_model.SaveOptions() is not None
    tf.saved_model.contains_saved_model(sm_dir)
    assert tf.saved_model.experimental.Fingerprint() is not None
    assert tf.saved_model.experimental.TrackableResource() is not None
    assert tf.saved_model.experimental.VariablePolicy() is not None
    assert tf.saved_model.experimental.read_fingerprint(sm_dir) is None


def test_signal_stubs() -> None:
    """Verify `tf.signal` FFT, windowing, and STFT operations."""
    f = tf.signal.fft([1.0, 2.0, 3.0, 4.0])
    assert len(f) == 4
    i = tf.signal.ifft(f)
    assert len(i) == 4
    rf = tf.signal.rfft([1.0, 2.0, 3.0, 4.0], fft_length=[4])
    assert len(rf) == 3
    irf = tf.signal.irfft(rf, fft_length=[4])
    assert len(irf) == 4
    f2d = tf.signal.fft2d([[1.0, 2.0], [3.0, 4.0]])
    assert f2d.shape == (2, 2)
    f3d = tf.signal.fft3d([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    assert f3d.shape == (2, 2, 2)

    h = tf.signal.hann_window(10, periodic=True)
    h_sym = tf.signal.hann_window(10, periodic=False)
    assert len(h) == 10 and len(h_sym) == 10
    assert not np.allclose(h.numpy(), h_sym.numpy())
    assert tf.signal.hann_window(0).shape == (0,)
    assert tf.signal.hann_window(1).numpy() == [1.0]

    hm = tf.signal.hamming_window(10, periodic=True)
    hm_sym = tf.signal.hamming_window(10, periodic=False)
    assert len(hm) == 10 and len(hm_sym) == 10
    assert not np.allclose(hm.numpy(), hm_sym.numpy())
    assert tf.signal.hamming_window(0).shape == (0,)
    assert tf.signal.hamming_window(1).numpy() == [1.0]

    s = tf.signal.stft([1.0, 2.0, 3.0, 4.0], frame_length=2, frame_step=1)
    assert isinstance(s, tf.Tensor)


def test_strings_stubs() -> None:
    """Verify `tf.strings` operations including ngrams, unicode, length units, and numeric parsing."""
    assert tf.strings.join(["a", "b"]) == "ab"
    assert tf.strings.join(["a"]) == "a"
    assert tf.strings.join([]) == ""
    assert tf.strings.regex_replace("abc", "b", "d") == "adc"
    assert tf.strings.regex_replace("abc", "b", "d", replace_global=False) == "adc"
    assert tf.strings.split("a b c", " ") == ["a", "b", "c"]
    assert tf.strings.split(["a b", "c d"], " ") == [["a", "b"], ["c", "d"]]
    assert tf.strings.bytes_split("abc") == ["a", "b", "c"]
    assert tf.strings.bytes_split(["ab", "cd"]) == [["a", "b"], ["c", "d"]]

    class MockTensor:
        """Test helper implementing `NumpyConvertibleProtocol`."""

        def numpy(self) -> TensorOperand:
            """Return a 0-D NumPy string array."""
            return np.array("a")

    assert tf.strings.length(MockTensor()) == 1
    assert tf.strings.length("é", unit="BYTE") == 2
    assert tf.strings.length("é", unit="UTF8_CHAR") == 1
    assert tf.strings.strip(" a") == "a"
    assert tf.strings.lower("A") == "a"
    assert tf.strings.upper("a") == "A"
    assert tf.strings.substr("abc", 1, 1) == "b"
    assert tf.strings.regex_full_match("abc", ".*b.*") is True
    n_float = tf.strings.to_number("1.2")
    assert isinstance(n_float, float) and np.isclose(n_float, 1.2)
    n_int = tf.strings.to_number("42", out_type=tf.int32)
    assert n_int == 42
    assert tf.strings.format("{} {}", ["a", "b"]) == "a b"
    assert tf.strings.reduce_join(["a", "b"], separator="-") == "a-b"
    hb = tf.strings.to_hash_bucket("a", 10)
    assert isinstance(hb, int) and 0 <= hb < 10
    assert tf.strings.to_hash_bucket_fast("a", 10) == hb
    assert tf.strings.to_hash_bucket_strong("a", 10, [1, 2]) == hb

    assert tf.strings.ngrams(["a", "b", "c"], 2) == ["a b", "b c"]
    assert tf.strings.ngrams(["a"], 2, preserve_short_sequences=True) == ["a"]
    assert tf.strings.ngrams(["a", "b"], 2, pad_values=("L", "R")) == ["L a", "a b", "b R"]
    assert tf.strings.ngrams([], 2) == []

    assert tf.strings.unicode_decode("ab", "utf-8") == [97, 98]
    codes, offsets = tf.strings.unicode_decode_with_offsets("ab", "utf-8")
    assert codes == [97, 98] and offsets == [0, 1]
    assert tf.strings.unicode_encode([97, 98], "utf-8") == ["ab"]
    assert tf.strings.unicode_script([97, 98]) == [0, 0]
    assert tf.strings.unicode_split("ab", "utf-8") == ["a", "b"]
    chars, char_offsets = tf.strings.unicode_split_with_offsets("ab", "utf-8")
    assert chars == ["a", "b"] and char_offsets == [0, 1]
    assert tf.strings.unicode_transcode("hello", "utf-8", "utf-8") == ["hello"]
    assert tf.strings.unsorted_segment_join(["a", "b", "c"], [0, 1, 0], 2, separator="-") == [
        "a-c",
        "b",
    ]


def test_train_stubs() -> None:
    """Verify `tf.train` checkpoint saving, `CheckpointManager`, `ExponentialMovingAverage`, and inspection functions."""
    v = tf.Variable(10.0)
    ckpt = tf.train.Checkpoint(var=v)
    saved_prefix = ckpt.save("dir/my_ckpt")
    assert saved_prefix == "dir/my_ckpt"
    v.assign(99.0)
    status = ckpt.restore("dir/my_ckpt")
    assert status is not None
    assert v.numpy() == 10.0

    manager = tf.train.CheckpointManager(ckpt, "mgr_dir", max_to_keep=2)
    p1 = manager.save()
    p2 = manager.save()
    p3 = manager.save()
    assert p1 == "mgr_dir/ckpt-1"
    assert p2 == "mgr_dir/ckpt-2"
    assert p3 == "mgr_dir/ckpt-3"
    assert manager.latest_checkpoint == "mgr_dir/ckpt-3"
    assert manager.checkpoints == ["mgr_dir/ckpt-2", "mgr_dir/ckpt-3"]
    assert tf.train.latest_checkpoint("mgr_dir") == "mgr_dir/ckpt-3"
    assert tf.train.list_variables("mgr_dir") == [("var", [])]
    assert tf.train.load_variable("mgr_dir", "var") == 10.0
    assert tf.train.load_checkpoint("mgr_dir") == {"var": 10.0}

    ema = tf.train.ExponentialMovingAverage(decay=0.5, num_updates=10)
    ema.apply([v])
    assert ema.average(v) == 10.0
    v.assign(20.0)
    ema.apply([v])
    assert ema.average(v) == 15.0
    assert ema.average_name(v) == "Variable/ExponentialMovingAverage"
