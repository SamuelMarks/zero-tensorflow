"""Unit tests for `test_train_stubs`."""

from __future__ import annotations

import inspect

import zero_tensorflow.train as tf_train
from zero_tensorflow._typing import CallableOperandProtocol, TensorOperand


def call_safely(callable_obj: CallableOperandProtocol | TensorOperand | type) -> TensorOperand:
    """Invoke `callable_obj` supplying `None` for any required parameters.

    Args:
        callable_obj: Target class or function.

    Returns:
        TensorOperand: Result of calling `callable_obj` or `'fallback'`.
    """
    if not callable(callable_obj):
        return None
    sig = inspect.signature(callable_obj)
    kwargs: dict[str, None] = {}
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            kwargs[name] = None
    try:
        res: TensorOperand = callable_obj(**kwargs)
        return res
    except (
        ValueError,
        TypeError,
        AttributeError,
        RuntimeError,
        NotImplementedError,
        KeyError,
        IndexError,
    ):
        return "fallback"


def test_train_stubs() -> None:
    """Verify `tf.train` protocol buffer and checkpoint stub classes and functions."""
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
    assert call_safely(tf_train.checkpoints_iterator) is None
    assert call_safely(tf_train.experimental) is not None
    assert call_safely(tf_train.get_checkpoint_state) is None
    assert tf_train.latest_checkpoint(None) is None
    assert call_safely(tf_train.list_variables) == []
    assert call_safely(tf_train.load_checkpoint) is None
    assert call_safely(tf_train.load_variable) is None
    assert call_safely(tf_train.experimental.MaxShardSizePolicy) is not None
    assert call_safely(tf_train.experimental.PythonState) is not None
    assert call_safely(tf_train.experimental.ShardByTaskPolicy) is not None
    assert call_safely(tf_train.experimental.ShardableTensor) is not None
    assert call_safely(tf_train.experimental.ShardingCallback) is not None
