"""Unit tests for `test_coverage_dtypes`."""

from __future__ import annotations

import contextlib

from zero_tensorflow import Tensor, _TracingContext

import numpy as np


def test_tensor_dtypes() -> None:
    """Execute `test_tensor_dtypes`."""
    Tensor(np.array(1.0, dtype=np.float64))
    Tensor(np.array(1.0, dtype=np.float32))
    Tensor(np.array(1, dtype=np.int64))
    Tensor(np.array(1, dtype=np.int32))
    Tensor(np.array(True, dtype=bool))
    Tensor(1.0, dtype="float32")
    Tensor(np.array(1, dtype=np.int8))
    Tensor(np.array("test string"))
    from zero_tensorflow import _to_tensor

    from ml_switcheroo_compiler.tracing import ProxyTensor

    pt = ProxyTensor("test", (), "float32")
    _to_tensor(pt)
    pt_no_dtype = ProxyTensor("test", ())
    _to_tensor(pt_no_dtype)
    pt_invalid_dtype = ProxyTensor("test", (), "invalid_dtype_string")
    _to_tensor(pt_invalid_dtype)
    _to_tensor(np.array(1, dtype=np.uint64))


def test_bool_errors() -> None:
    """Execute `test_bool_errors`."""
    from ml_switcheroo_ir import LogicalNode

    t_multi = Tensor([1, 2])
    with contextlib.suppress(ValueError):
        bool(t_multi)

    t_traced = Tensor(1.0, _traced_node=LogicalNode(id="test_id", op_type="Custom"))
    with contextlib.suppress(TypeError):
        bool(t_traced)


def test_tracing_context() -> None:
    """Execute `test_tracing_context`."""
    _ = _TracingContext()
    _TracingContext.enter()
    _TracingContext.exit()
    _TracingContext.get()
