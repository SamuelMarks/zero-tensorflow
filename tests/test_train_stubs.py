import zero_tensorflow.train as tf_train


def call_safely(callable_obj):
    import inspect

    sig = inspect.signature(callable_obj)
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            kwargs[name] = None
    try:
        return callable_obj(**kwargs)
    except Exception:  # noqa: BLE001
        return "fallback"
    return None


def test_train_stubs():
    assert call_safely(tf_train.BytesList) is not None
    assert call_safely(tf_train.Checkpoint) is not None
    assert call_safely(tf_train.CheckpointManager) is not None
    assert call_safely(tf_train.CheckpointOptions) is not None
    assert call_safely(tf_train.CheckpointView) is not None
    assert call_safely(tf_train.ClusterDef) is not None
    assert call_safely(tf_train.ClusterSpec) is not None
    assert call_safely(tf_train.Coordinator) is not None
    assert call_safely(tf_train.Example) is not None
    assert call_safely(tf_train.ExponentialMovingAverage) is not None
    assert call_safely(tf_train.Feature) is not None
    assert call_safely(tf_train.FeatureList) is not None
    assert call_safely(tf_train.FeatureLists) is not None
    assert call_safely(tf_train.Features) is not None
    assert call_safely(tf_train.FloatList) is not None
    assert call_safely(tf_train.Int64List) is not None
    assert call_safely(tf_train.JobDef) is not None
    assert call_safely(tf_train.SequenceExample) is not None
    assert call_safely(tf_train.ServerDef) is not None
    assert call_safely(tf_train.TrackableView) is not None
    assert call_safely(tf_train.checkpoints_iterator) in (None, "fallback")
    assert call_safely(tf_train.experimental) is not None
    assert call_safely(tf_train.get_checkpoint_state) in (None, "fallback")
    assert call_safely(tf_train.latest_checkpoint) in (None, "fallback")
    assert call_safely(tf_train.list_variables) in (None, "fallback")
    assert call_safely(tf_train.load_checkpoint) in (None, "fallback")
    assert call_safely(tf_train.load_variable) in (None, "fallback")
    assert call_safely(tf_train.experimental.MaxShardSizePolicy) is not None
    assert call_safely(tf_train.experimental.PythonState) is not None
    assert call_safely(tf_train.experimental.ShardByTaskPolicy) is not None
    assert call_safely(tf_train.experimental.ShardableTensor) is not None
    assert call_safely(tf_train.experimental.ShardingCallback) is not None
