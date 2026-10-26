"""Unit tests for `test_coverage_dtypes`."""

from __future__ import annotations

import contextlib

from zero_tensorflow import Tensor, _to_tensor, _TracingContext

from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_ir import LogicalNode


def test_tensor_dtypes() -> None:
    """Execute `test_tensor_dtypes`."""
    Tensor(1.0, dtype="float64")
    Tensor(1.0, dtype="float32")
    Tensor(1, dtype="int64")
    Tensor(1, dtype="int32")
    Tensor(True, dtype="bool")
    Tensor(1.0, dtype="float32")
    Tensor(1, dtype="int8")
    Tensor("test string")

    pt = ProxyTensor("test", (), "float32")
    _to_tensor(pt)
    pt_no_dtype = ProxyTensor("test", ())
    _to_tensor(pt_no_dtype)
    pt_invalid_dtype = ProxyTensor("test", (), "invalid_dtype_string")
    _to_tensor(pt_invalid_dtype)
    _to_tensor(Tensor(1, dtype="uint64"))


def test_bool_errors() -> None:
    """Execute `test_bool_errors`."""
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
