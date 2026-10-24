"""Unit tests for `test_types_stubs`."""

from __future__ import annotations

import zero_tensorflow.types as tf_types
from zero_tensorflow._typing import CallableOperandProtocol, TensorOperand


def call_safely(callable_obj: CallableOperandProtocol | TensorOperand | type) -> TensorOperand:
    """Invoke `callable_obj` safely with zero or one `None` argument.

    Args:
        callable_obj: Target class or function.

    Returns:
        TensorOperand: Result of calling `callable_obj` or `None`.
    """
    if not callable(callable_obj):
        return None
    try:
        res: TensorOperand = callable_obj()
        return res
    except TypeError:
        try:
            res2: TensorOperand = callable_obj(None)
            return res2
        except TypeError:
            return None


def test_types_stubs() -> None:
    """Verify `tf.types.experimental` stub classes and functions."""
    assert call_safely(tf_types.experimental) is not None
    assert call_safely(tf_types.experimental.AtomicFunction) is not None
    assert call_safely(tf_types.experimental.Callable) is not None
    assert call_safely(tf_types.experimental.ConcreteFunction) is not None
    assert call_safely(tf_types.experimental.FunctionType) is not None
    assert call_safely(tf_types.experimental.GenericFunction) is not None
    assert call_safely(tf_types.experimental.PolymorphicFunction) is not None
    assert call_safely(tf_types.experimental.SupportsTracingProtocol) is not None
    assert call_safely(tf_types.experimental.TensorLike) is not None
    assert call_safely(tf_types.experimental.TraceType) is not None
    assert call_safely(tf_types.experimental.distributed) is None
