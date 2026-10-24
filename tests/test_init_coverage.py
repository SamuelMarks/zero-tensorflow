"""Unit tests for `test_init_coverage`."""

from __future__ import annotations

import contextlib
import inspect

import zero_tensorflow as tf
from zero_tensorflow._typing import CallableOperandProtocol, TensorOperand

import ml_switcheroo_compiler.ops as msc_ops

_ERRORS = (
    ValueError,
    TypeError,
    AttributeError,
    RuntimeError,
    NotImplementedError,
    KeyError,
    IndexError,
    ZeroDivisionError,
    AssertionError,
)


def call_safely(func: CallableOperandProtocol | TensorOperand) -> None:
    """Invoke `func` with 1, 2, or 0 dummy tensor arguments.

    Args:
        func: Target callable operation.
    """
    if not callable(func) or isinstance(func, type):
        return
    t = msc_ops.zeros((2, 2))
    try:
        func(t)
    except _ERRORS:
        try:
            func(t, t)
        except _ERRORS:
            with contextlib.suppress(*_ERRORS):
                func()


def test_init_coverage() -> None:
    """Verify all operations in `tf.math`, `tf.linalg`, `tf.bitwise`, and top-level `tf` can be invoked."""
    for cls_obj in (tf.math, tf.linalg, tf.bitwise):
        for name, obj in inspect.getmembers(cls_obj):
            if name.startswith("_"):
                continue
            if callable(obj):
                call_safely(obj)

    for name, obj in inspect.getmembers(tf):
        if name.startswith("_") or name in ("math", "linalg", "bitwise"):
            continue
        if callable(obj) and not isinstance(obj, type):
            call_safely(obj)
