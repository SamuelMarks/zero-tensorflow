"""Unit tests for `test_train_stubs`."""

from __future__ import annotations

from pathlib import Path

import zero_tensorflow as tf
import zero_tensorflow.train as tf_train


def test_train_stubs(tmp_path: Path) -> None:
    """Verify `tf.train` protocol buffer containers, `ClusterSpec`, `Coordinator`, and disk checkpoints.

    Args:
        tmp_path: Temporary directory fixture.
    """
    bl = tf_train.BytesList(value=[b"hello", "world"])
    assert bl.value == [b"hello", b"world"]
    assert tf_train.BytesList().value == []

    fl = tf_train.FloatList(value=[1.5, 2])
    assert fl.value == [1.5, 2.0]
    assert tf_train.FloatList().value == []

    il = tf_train.Int64List(value=[10, 20])
    assert il.value == [10, 20]
    assert tf_train.Int64List().value == []

    feat = tf_train.Feature(bytes_list=bl, float_list=fl, int64_list=il)
    features = tf_train.Features(feature={"a": feat})
    ex = tf_train.Example(features=features)
    assert ex.features.feature["a"].int64_list.value == [10, 20]
    raw_ex = ex.SerializeToString()
    assert b"hello" in raw_ex
    assert tf_train.Example().features.feature == {}

    flist = tf_train.FeatureList(feature=[feat])
    assert len(flist.feature) == 1
    assert tf_train.FeatureList().feature == []
    flists = tf_train.FeatureLists(feature_list={"seq": flist})
    seq_ex = tf_train.SequenceExample(context=features, feature_lists=flists)
    assert "seq" in seq_ex.feature_lists.feature_list
    assert tf_train.SequenceExample().context.feature == {}
    assert tf_train.FeatureLists().feature_list == {}

    jdef1 = tf_train.JobDef(name="worker", tasks={0: "localhost:2222", 1: "localhost:2223"})
    jdef2 = tf_train.JobDef(name="ps", tasks=["localhost:2224"])
    assert tf_train.JobDef().tasks == {}
    cdef = tf_train.ClusterDef(job=[jdef1, jdef2])
    assert len(cdef.job) == 2
    assert tf_train.ClusterDef().job == []

    cspec_from_def = tf_train.ClusterSpec(cdef)
    assert cspec_from_def.jobs == ["ps", "worker"]
    assert cspec_from_def.num_tasks("worker") == 2
    assert cspec_from_def.task_address("worker", 1) == "localhost:2223"

    cspec_from_spec = tf_train.ClusterSpec(cspec_from_def)
    assert cspec_from_spec.as_dict() == cspec_from_def.as_dict()

    cspec_from_dict = tf_train.ClusterSpec({"worker": ["w0:1", "w1:1"], "ps": {0: "ps0:1"}})
    assert cspec_from_dict.num_tasks("ps") == 1
    assert tf_train.ClusterSpec().as_dict() == {}

    coord = tf_train.Coordinator(clean_stop_exception_types=[ValueError])
    assert coord.should_stop() is False
    coord.request_stop()
    assert coord.should_stop() is True
    coord.clear_stop()
    assert coord.should_stop() is False
    coord.join()
    assert tf_train.Coordinator().should_stop() is False

    opts = tf_train.CheckpointOptions(
        experimental_io_device="/job:localhost",
        experimental_enable_async_checkpoint=True,
    )
    assert opts.experimental_io_device == "/job:localhost"
    assert opts.experimental_enable_async_checkpoint is True

    sdef = tf_train.ServerDef(cluster=cdef, job_name="worker", task_index=1)
    assert sdef.job_name == "worker" and sdef.task_index == 1

    # Disk-persisted Checkpoint and CheckpointManager round-trip
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    v = tf.Variable([1.0, 2.0])
    s_var = tf.Variable(5.0)
    ckpt = tf_train.Checkpoint(vec=v, scalar=s_var)
    mgr = tf_train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=1)
    p1 = mgr.save()
    p2 = mgr.save()
    assert not (ckpt_dir / "ckpt-1.ckpt.json").exists()
    assert (ckpt_dir / "ckpt-2.ckpt.json").exists()
    assert (ckpt_dir / "checkpoint").exists()
    assert p1 != p2

    # Clear in-memory store to verify disk restoration and inspection
    tf_train._CHECKPOINT_STORE.clear()
    tf_train._DIRECTORY_LATEST.clear()
    tf_train._DIRECTORY_ALL.clear()

    state = tf_train.get_checkpoint_state(str(ckpt_dir))
    assert state is not None
    assert state.model_checkpoint_path == p2
    assert tf_train.checkpoints_iterator(str(ckpt_dir)) == [p2]
    assert tf_train.latest_checkpoint(str(ckpt_dir)) == p2
    assert tf_train.list_variables(str(ckpt_dir)) == [("scalar", []), ("vec", [2])]
    loaded_dict = tf_train.load_checkpoint(str(ckpt_dir))
    assert loaded_dict is not None and "vec" in loaded_dict
    assert tf_train.load_variable(p2, "scalar") == 5.0

    v.assign([0.0, 0.0])
    s_var.assign(0.0)
    restore_status = ckpt.restore(p2)
    assert restore_status is not None
    tf.debugging.assert_near(v, [1.0, 2.0])
    assert s_var.numpy() == 5.0

    view = tf_train.CheckpointView(p2)
    assert view.children() == {"scalar": 1, "vec": 2}
    assert tf_train.CheckpointView().children() == {}

    class _DummyTrackableRoot:
        """Trackable container for testing `TrackableView.children()`."""

        @property
        def variables(self) -> list[tf.Variable]:
            """Return tracked variables.

            Returns:
                list[tf.Variable]: Single variable list.
            """
            return [s_var]

    assert "var_0" in tf_train.TrackableView(_DummyTrackableRoot()).children()
    assert tf_train.TrackableView().children() == {}

    # In-memory get_checkpoint_state fallback when directory does not exist on disk
    ckpt.save("synthetic_mem_dir/ckpt-1")
    mem_state = tf_train.get_checkpoint_state("synthetic_mem_dir")
    assert mem_state is not None and mem_state.model_checkpoint_path == "synthetic_mem_dir/ckpt-1"
    assert tf_train.checkpoints_iterator(None) == []
    assert tf_train.get_checkpoint_state(None) is None
    assert tf_train.get_checkpoint_state("nonexistent_dir") is None
    assert tf_train.latest_checkpoint(None) is None
    assert tf_train.list_variables(None) == []
    assert tf_train.load_checkpoint(None) is None
    assert tf_train.load_variable(None, None) is None

    # ExponentialMovingAverage with zero_debias=True
    ema = tf_train.ExponentialMovingAverage(decay=0.5, zero_debias=True)
    ema_var = tf.Variable(10.0)
    ema.apply([ema_var])
    assert ema.average(ema_var) == 10.0
    ema_var.assign(20.0)
    ema.apply([ema_var])
    # Step 2: biased = 0.5 * 5.0 + 0.5 * 20.0 = 12.5; denom = 1 - 0.25 = 0.75 -> 12.5 / 0.75
    avg_step2 = ema.average(ema_var)
    assert isinstance(avg_step2, float)
    tf.debugging.assert_near(avg_step2, 12.5 / 0.75)

    p_policy = tf_train.experimental.MaxShardSizePolicy(1024)
    assert p_policy.max_shard_size == 1024
    py_state = tf_train.experimental.PythonState()
    py_state.deserialize("custom_payload")
    assert py_state.serialize() == "custom_payload"
    assert tf_train.experimental.ShardByTaskPolicy(4).num_shards == 4
    st = tf_train.experimental.ShardableTensor(tensor=v, shape=(2,))
    assert st.shape == (2,)
    assert tf_train.experimental.ShardingCallback("desc").description == "desc"
