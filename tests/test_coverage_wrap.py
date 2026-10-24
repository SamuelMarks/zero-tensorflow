"""Unit tests for `test_coverage_wrap`."""

from __future__ import annotations

from zero_tensorflow import Tensor, _to_tensor, _wrap, _wrap_tensor, function
from zero_tensorflow._typing import TensorOperand

import ml_switcheroo_compiler
import pytest
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_ir import LogicalGraph, LogicalNode


def test_wrap_and_to_tensor_branches() -> None:
    """Verify `_wrap`, `_wrap_tensor`, and `_to_tensor` across eager and traced execution."""
    t = Tensor(1.0)
    assert _wrap(t) is t
    assert _wrap_tensor(t) is t
    assert _wrap_tensor(2.0).numpy() == 2.0

    wrapped_tuple = _wrap((1.0, 2.0))
    assert isinstance(wrapped_tuple, tuple) and len(wrapped_tuple) == 2
    assert all(isinstance(item, Tensor) for item in wrapped_tuple)

    wrapped_list = _wrap([1.0, 2.0])
    assert isinstance(wrapped_list, list) and len(wrapped_list) == 2
    assert all(isinstance(item, Tensor) for item in wrapped_list)

    node = LogicalNode(id="traced_n1", op_type="Relu")
    t_from_node = Tensor(_traced_node=node)
    assert t_from_node.shape == ()
    with pytest.raises(ValueError, match="Cannot call array conversion on a traced tensor"):
        t_from_node.numpy()

    t_retyped = Tensor(t, dtype="float64")
    assert t_retyped.dtype == DType.Float64

    t_none = Tensor(None)
    assert t_none.shape == ()
    assert t_none.dtype is None
    assert t_none.numpy() is None

    prev_tr = _tracer.is_tracing
    prev_gr = _tracer.active_graph
    _tracer.is_tracing = True
    g_const = LogicalGraph(name="const_non_num")
    _tracer.active_graph = g_const
    try:
        n_bad = LogicalNode(id="c_str", op_type="Constant", attributes={"value": "text"})
        g_const.nodes["c_str"] = n_bad
        t_c_str = Tensor(_traced_node=n_bad)
        with pytest.raises(ValueError, match="Cannot call array conversion on a traced tensor"):
            t_c_str.numpy()
    finally:
        _tracer.is_tracing = prev_tr
        _tracer.active_graph = prev_gr

    pt = ProxyTensor(id="pt_1", shape=(2,), dtype="float32")
    t_from_pt = _to_tensor(pt)
    assert isinstance(t_from_pt, ml_switcheroo_compiler.Tensor)

    raw_msc = _to_tensor(3.0)

    @function
    def traced_fn() -> tuple[TensorOperand, TensorOperand, TensorOperand]:
        """Trace repeated uses of `t` and raw `ml_switcheroo_compiler.Tensor`.

        Returns:
            tuple[TensorOperand, TensorOperand, TensorOperand]: Traced tensor pair and raw tensor.
        """
        first = _to_tensor(t)
        second = _to_tensor(t)
        third = _to_tensor(raw_msc)
        return (first, second, third)

    r1, r2, r3 = traced_fn()
    assert isinstance(r1, ml_switcheroo_compiler.Tensor)
    assert isinstance(r2, ml_switcheroo_compiler.Tensor)
    assert isinstance(r3, ml_switcheroo_compiler.Tensor)

    prev_tracing = _tracer.is_tracing
    prev_graph = _tracer.active_graph
    try:
        _tracer.is_tracing = True
        g = LogicalGraph(name="const_lookup")
        _tracer.active_graph = g
        traced_const = Tensor(_to_tensor(Tensor(7.0)))
        assert traced_const.numpy() == 7.0
        unknown_pt = ProxyTensor(id="missing_id", shape=(), dtype="float32")
        t_unknown = Tensor(
            ml_switcheroo_compiler.Tensor(
                data=unknown_pt,
                config=TensorConfig(shape=(), dtype=DType.Float32, device="cpu"),
            )
        )
        with pytest.raises(ValueError, match="Cannot call array conversion on a traced tensor"):
            t_unknown.numpy()
    finally:
        _tracer.is_tracing = prev_tracing
        _tracer.active_graph = prev_graph
