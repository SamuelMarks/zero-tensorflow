"""Unit tests for `test_data_stubs`."""

from __future__ import annotations

import contextlib
import inspect

import zero_tensorflow.data as tf_data
from zero_tensorflow._typing import CallableOperandProtocol, TensorOperand


def call_with_dummies(func: CallableOperandProtocol | TensorOperand | type) -> None:
    """Invoke `func` passing `None` for required parameters.

    Args:
        func: Target `tf.data` class or function.
    """
    if not callable(func):
        return
    sig = inspect.signature(func)
    kwargs: dict[str, None] = {}
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            kwargs[name] = None
    with contextlib.suppress(
        ValueError,
        TypeError,
        AttributeError,
        RuntimeError,
        NotImplementedError,
        KeyError,
        IndexError,
    ):
        func(**kwargs)


def test_data_stubs() -> None:
    """Verify all public classes and functions on `tf.data` and `tf.data.experimental`."""
    for name, obj in inspect.getmembers(tf_data):
        if name.startswith("_"):
            continue
        if callable(obj):
            call_with_dummies(obj)

    for name, obj in inspect.getmembers(tf_data.experimental):
        if name.startswith("_"):
            continue
        if callable(obj):
            call_with_dummies(obj)
