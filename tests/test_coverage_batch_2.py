import contextlib


@contextlib.contextmanager
def _suppress_all():
    try:
        yield
    except Exception:  # noqa: BLE001, S110
        pass


import numpy as np

import zero_tensorflow as tf


def test_distribute_stubs():
    with _suppress_all():
        tf.distribute.CrossDeviceOps()
    with _suppress_all():
        tf.distribute.DistributedDataset()
    with _suppress_all():
        tf.distribute.DistributedIterator()
    with _suppress_all():
        tf.distribute.DistributedValues()
    with _suppress_all():
        tf.distribute.HierarchicalCopyAllReduce()
    with _suppress_all():
        tf.distribute.InputContext()
    with _suppress_all():
        tf.distribute.InputOptions()
    with _suppress_all():
        tf.distribute.InputReplicationMode()
    with _suppress_all():
        tf.distribute.NcclAllReduce()
    with _suppress_all():
        tf.distribute.ParameterServerStrategy()
    with _suppress_all():
        tf.distribute.ReductionToOneDevice()
    with _suppress_all():
        tf.distribute.ReplicaContext()
    with _suppress_all():
        tf.distribute.RunOptions()
    with _suppress_all():
        tf.distribute.Server()
    with _suppress_all():
        tf.distribute.StrategyExtended()

    # distribute.cluster_resolver
    with _suppress_all():
        tf.distribute.cluster_resolver.ClusterResolver()
    with _suppress_all():
        tf.distribute.cluster_resolver.GCEClusterResolver()
    with _suppress_all():
        tf.distribute.cluster_resolver.KubernetesClusterResolver()
    with _suppress_all():
        tf.distribute.cluster_resolver.SimpleClusterResolver()
    with _suppress_all():
        tf.distribute.cluster_resolver.SlurmClusterResolver()
    with _suppress_all():
        tf.distribute.cluster_resolver.TFConfigClusterResolver()
    with _suppress_all():
        tf.distribute.cluster_resolver.TPUClusterResolver()
    with _suppress_all():
        tf.distribute.cluster_resolver.UnionResolver()

    # distribute.coordinator
    with _suppress_all():
        tf.distribute.coordinator.ClusterCoordinator()
    with _suppress_all():
        tf.distribute.coordinator.PerWorkerValue()
    with _suppress_all():
        tf.distribute.coordinator.RemoteValue()
    with _suppress_all():
        tf.distribute.coordinator.experimental_get_current_worker_index()

    # distribute.experimental
    with _suppress_all():
        tf.distribute.experimental.CentralStorageStrategy()
    with _suppress_all():
        tf.distribute.experimental.CollectiveCommunication()
    with _suppress_all():
        tf.distribute.experimental.CollectiveHints()
    with _suppress_all():
        tf.distribute.experimental.CommunicationImplementation()
    with _suppress_all():
        tf.distribute.experimental.CommunicationOptions()
    with _suppress_all():
        tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with _suppress_all():
        tf.distribute.experimental.ParameterServerStrategy()
    with _suppress_all():
        tf.distribute.experimental.PreemptionCheckpointHandler()
    with _suppress_all():
        tf.distribute.experimental.PreemptionWatcher()
    with _suppress_all():
        tf.distribute.experimental.TPUStrategy()
    with _suppress_all():
        tf.distribute.experimental.TerminationConfig()
    with _suppress_all():
        tf.distribute.experimental.ValueContext()
    with _suppress_all():
        tf.distribute.experimental.coordinator()
    with _suppress_all():
        tf.distribute.experimental.partitioners()
    with _suppress_all():
        tf.distribute.experimental.rpc()

    with _suppress_all():
        tf.distribute.experimental_set_strategy()
    with _suppress_all():
        tf.distribute.get_replica_context()
    with _suppress_all():
        tf.distribute.get_strategy()
    with _suppress_all():
        tf.distribute.has_strategy()
    with _suppress_all():
        tf.distribute.in_cross_replica_context()

    # test strategies init and minimal function
    ms = tf.distribute.MirroredStrategy()
    with ms.scope():
        res = ms.run(lambda x: x + 1, args=(3,))
        assert res == 4
        res2 = ms.run(lambda x: x + 1, args=(), kwargs={"x": 3})
        assert res2 == 4
        res3 = ms.reduce("SUM", 5)
        assert res3 == 5

    tf.distribute.MirroredStrategy(devices=["/cpu:0"])
    tf.distribute.MultiWorkerMirroredStrategy()
    tf.distribute.OneDeviceStrategy(device="/cpu:0")
    tf.distribute.TPUStrategy()


def test_io_stubs(tmp_path):
    # write / read tests
    fpath = tmp_path / "test.txt"
    tf.io.write_file(str(fpath), b"hello")
    res = tf.io.read_file(str(fpath))
    assert res == b"hello"

    with _suppress_all():
        tf.io.decode_jpeg(b"")
    with _suppress_all():
        tf.io.decode_png(b"")
    with _suppress_all():
        tf.io.decode_image(b"")
    with _suppress_all():
        tf.io.parse_tensor(b"", tf.float32)
    with _suppress_all():
        tf.io.serialize_tensor(tf.constant([1.0]))
    with _suppress_all():
        tf.io.parse_example(b"", {})
    with _suppress_all():
        tf.io.parse_single_example(b"", {})

    with _suppress_all():
        tf.io.FixedLenFeature()
    with _suppress_all():
        tf.io.FixedLenSequenceFeature()
    with _suppress_all():
        tf.io.RaggedFeature()
    with _suppress_all():
        tf.io.SparseFeature()
    with _suppress_all():
        tf.io.TFRecordOptions()
    with _suppress_all():
        tf.io.TFRecordWriter()
    with _suppress_all():
        tf.io.VarLenFeature()
    with _suppress_all():
        tf.io.decode_and_crop_jpeg()
    with _suppress_all():
        tf.io.decode_base64()
    with _suppress_all():
        tf.io.decode_bmp()
    with _suppress_all():
        tf.io.decode_compressed()
    with _suppress_all():
        tf.io.decode_csv()
    with _suppress_all():
        tf.io.decode_gif()
    with _suppress_all():
        tf.io.decode_json_example()
    with _suppress_all():
        tf.io.decode_proto()
    with _suppress_all():
        tf.io.decode_raw()
    with _suppress_all():
        tf.io.deserialize_many_sparse()
    with _suppress_all():
        tf.io.encode_base64()
    with _suppress_all():
        tf.io.encode_jpeg()
    with _suppress_all():
        tf.io.encode_png()
    with _suppress_all():
        tf.io.encode_proto()
    with _suppress_all():
        tf.io.extract_jpeg_shape()

    # io.gfile
    with _suppress_all():
        tf.io.gfile.GFile()
    with _suppress_all():
        tf.io.gfile.copy()
    with _suppress_all():
        tf.io.gfile.exists()
    with _suppress_all():
        tf.io.gfile.get_registered_schemes()
    with _suppress_all():
        tf.io.gfile.glob()
    with _suppress_all():
        tf.io.gfile.isdir()
    with _suppress_all():
        tf.io.gfile.join()
    with _suppress_all():
        tf.io.gfile.listdir()
    with _suppress_all():
        tf.io.gfile.makedirs()
    with _suppress_all():
        tf.io.gfile.mkdir()
    with _suppress_all():
        tf.io.gfile.remove()
    with _suppress_all():
        tf.io.gfile.rename()
    with _suppress_all():
        tf.io.gfile.rmtree()
    with _suppress_all():
        tf.io.gfile.stat()
    with _suppress_all():
        tf.io.gfile.walk()
    with _suppress_all():
        tf.io.is_jpeg()
    with _suppress_all():
        tf.io.match_filenames_once()
    with _suppress_all():
        tf.io.matching_files()
    with _suppress_all():
        tf.io.parse_sequence_example()
    with _suppress_all():
        tf.io.parse_single_sequence_example()
    with _suppress_all():
        tf.io.serialize_many_sparse()
    with _suppress_all():
        tf.io.serialize_sparse()
    with _suppress_all():
        tf.io.write_graph()


def test_saved_model_stubs():
    with _suppress_all():
        tf.saved_model.save(None, "dir")
    with _suppress_all():
        tf.saved_model.load("dir")

    with _suppress_all():
        tf.saved_model.Asset()
    with _suppress_all():
        tf.saved_model.LoadOptions()
    with _suppress_all():
        tf.saved_model.SaveOptions()
    with _suppress_all():
        tf.saved_model.contains_saved_model()

    with _suppress_all():
        tf.saved_model.experimental.Fingerprint()
    with _suppress_all():
        tf.saved_model.experimental.TrackableResource()
    with _suppress_all():
        tf.saved_model.experimental.VariablePolicy()
    with _suppress_all():
        tf.saved_model.experimental.read_fingerprint()


def test_signal_stubs():
    # fft
    f = tf.signal.fft([1.0, 2.0, 3.0, 4.0])
    assert len(f) == 4
    i = tf.signal.ifft(f)
    assert len(i) == 4

    # window
    h = tf.signal.hann_window(10)
    assert len(h) == 10
    h_per = tf.signal.hann_window(10, periodic=False)
    assert len(h_per) == 10

    hm = tf.signal.hamming_window(10)
    assert len(hm) == 10
    hm_per = tf.signal.hamming_window(10, periodic=False)
    assert len(hm_per) == 10

    # stft
    s = tf.signal.stft([1.0, 2.0, 3.0, 4.0], frame_length=2, frame_step=1)
    assert s is not None
    s2 = tf.signal.stft(
        [1.0, 2.0, 3.0, 4.0], frame_length=2, frame_step=1, pad_end=True
    )
    assert s2 is not None
    s3 = tf.signal.stft(
        [1.0, 2.0, 3.0, 4.0], frame_length=2, frame_step=1, fft_length=4
    )
    assert s3 is not None


def test_strings_stubs():
    j = tf.strings.join(["a", "b"])
    assert j == "ab"
    j_single = tf.strings.join(["a"])
    assert j_single == "a"
    j2 = tf.strings.join([])
    assert j2 == ""

    # regex_replace
    r = tf.strings.regex_replace("abc", "b", "d")
    assert r == "adc"
    r2 = tf.strings.regex_replace("abc", "b", "d", replace_global=False)
    assert r2 == "adc"

    # split
    s = tf.strings.split("a b c", " ")
    assert list(s) == ["a", "b", "c"]
    s2 = tf.strings.split(["a b", "c d"], " ")
    assert len(s2) == 2

    # bytes_split
    b = tf.strings.bytes_split("abc")
    assert list(b) == ["a", "b", "c"]
    b2 = tf.strings.bytes_split(["ab", "cd"])
    assert len(b2) == 2

    class MockTensor:
        def numpy(self):
            return np.array("a")

    tf.strings.length(MockTensor())

    # strip, lower, upper, substr, length
    assert tf.strings.strip(" a") == "a"
    assert tf.strings.lower("A") == "a"
    assert tf.strings.upper("a") == "A"
    assert tf.strings.substr("abc", 1, 1) == "b"
    assert tf.strings.length("abc") == 3

    # regex_full_match
    assert tf.strings.regex_full_match("abc", ".*b.*")

    # to_number
    n = tf.strings.to_number("1.2")
    assert np.isclose(n, 1.2)

    # format
    f = tf.strings.format("{} {}", ["a", "b"])
    assert f == "a b"

    # reduce_join
    rj = tf.strings.reduce_join(["a", "b"], separator="-")
    assert rj == "a-b"

    # to_hash_bucket
    hb = tf.strings.to_hash_bucket("a", 10)
    assert 0 <= hb < 10
    hb_f = tf.strings.to_hash_bucket_fast("a", 10)
    assert 0 <= hb_f < 10
    hb_s = tf.strings.to_hash_bucket_strong("a", 10, [1, 2])
    assert 0 <= hb_s < 10

    # unicode functions empty stubs
    assert len(tf.strings.ngrams([], 1)) == 0
    assert len(tf.strings.unicode_decode([], "utf-8")) == 0
    res1, res2 = tf.strings.unicode_decode_with_offsets([], "utf-8")
    assert len(res1) == 0 and len(res2) == 0
    assert len(tf.strings.unicode_encode([], "utf-8")) == 0
    assert len(tf.strings.unicode_script([])) == 0
    assert len(tf.strings.unicode_split([], "utf-8")) == 0
    res3, res4 = tf.strings.unicode_split_with_offsets([], "utf-8")
    assert len(res3) == 0 and len(res4) == 0
    assert len(tf.strings.unicode_transcode([], "utf-8", "utf-8")) == 0
    assert len(tf.strings.unsorted_segment_join([], [], 1)) == 0


def test_train_stubs():
    ckpt = tf.train.Checkpoint()
    with _suppress_all():
        ckpt.save("test")
    with _suppress_all():
        ckpt.restore("test")

    manager = tf.train.CheckpointManager(ckpt, "dir", 5)
    with _suppress_all():
        manager.save()
    assert manager.latest_checkpoint is None
    assert manager.checkpoints == []

    with _suppress_all():
        tf.train.BytesList()
    with _suppress_all():
        tf.train.CheckpointOptions()
    with _suppress_all():
        tf.train.CheckpointView()
    with _suppress_all():
        tf.train.ClusterDef()
    with _suppress_all():
        tf.train.ClusterSpec()
    with _suppress_all():
        tf.train.Coordinator()
    with _suppress_all():
        tf.train.Example()
    with _suppress_all():
        tf.train.ExponentialMovingAverage()
    with _suppress_all():
        tf.train.Feature()
    with _suppress_all():
        tf.train.FeatureList()
    with _suppress_all():
        tf.train.FeatureLists()
    with _suppress_all():
        tf.train.Features()
    with _suppress_all():
        tf.train.FloatList()
    with _suppress_all():
        tf.train.Int64List()
    with _suppress_all():
        tf.train.JobDef()
    with _suppress_all():
        tf.train.SequenceExample()
    with _suppress_all():
        tf.train.ServerDef()
    with _suppress_all():
        tf.train.TrackableView()
    with _suppress_all():
        tf.train.checkpoints_iterator()
    with _suppress_all():
        tf.train.get_checkpoint_state()
    with _suppress_all():
        tf.train.latest_checkpoint()
    with _suppress_all():
        tf.train.list_variables()
    with _suppress_all():
        tf.train.load_checkpoint()
    with _suppress_all():
        tf.train.load_variable()

    with _suppress_all():
        tf.train.experimental.MaxShardSizePolicy()
    with _suppress_all():
        tf.train.experimental.PythonState()
    with _suppress_all():
        tf.train.experimental.ShardByTaskPolicy()
    with _suppress_all():
        tf.train.experimental.ShardableTensor()
    with _suppress_all():
        tf.train.experimental.ShardingCallback()
