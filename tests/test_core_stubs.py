"""Unit tests for `zero_tensorflow.core_stubs`."""

from __future__ import annotations

import zero_tensorflow as tf
from zero_tensorflow.core_stubs import (
    _to_float,
    _to_int,
    _to_py_data,
    double,
    import_graph_def,
    init_scope,
    is_symbolic_tensor,
    is_tensor,
    ragged_fill_empty_rows,
    ragged_fill_empty_rows_grad,
    size,
    space_to_batch,
    space_to_batch_nd,
    string,
)

import ml_switcheroo_compiler.ops as msc_ops
import pytest
from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_ir import LogicalGraph, LogicalNode


class _FakeArrayLike:
    """Pure-Python test double implementing `ArrayLikeProtocol`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array shape.

        Returns:
            tuple[int, ...]: 1-D shape tuple.
        """
        return (2,)

    @property
    def size(self) -> int:
        """Return element count.

        Returns:
            int: Element count.
        """
        return 2

    def tolist(self) -> list[int]:
        """Return nested Python list.

        Returns:
            list[int]: List of elements.
        """
        return [1, 2]

    def item(self) -> int:
        """Return scalar item.

        Returns:
            int: First element.
        """
        return 1


def test_core_stubs() -> None:
    """Verify `double`, `import_graph_def`, `ragged_fill_empty_rows`, and `space_to_batch*`."""
    tensor = msc_ops.zeros((2,))
    assert double(tensor).shape == (2,)
    assert double().numpy() == 0.0
    assert import_graph_def(None) is None

    g = LogicalGraph(name="source_graph")
    n1 = LogicalNode(id="in1", op_type="Placeholder", attributes={"dtype": "float32"})
    n2 = LogicalNode(id="out1", op_type="Relu", attributes={})
    g.nodes["in1"] = n1
    g.nodes["out1"] = n2
    g.add_edge("in1", "out1")

    imported_g = import_graph_def(g, name="scope")
    assert isinstance(imported_g, LogicalGraph)
    assert "scope/in1" in imported_g.nodes
    assert "scope/out1" in imported_g.nodes
    assert len(imported_g.edges) == 1

    replacement_node = LogicalNode(id="rep", op_type="Const", attributes={"value": 1.0})
    ret_nodes = import_graph_def(
        g,
        input_map={"in1": replacement_node, "out1": tf.Tensor(1.0)},
        return_elements=["in1:0", "out1"],
        op_dict={"Placeholder": "Placeholder", "Relu": "Relu"},
    )
    assert isinstance(ret_nodes, list) and len(ret_nodes) == 2
    assert ret_nodes[0].op_type == "Const"
    assert ret_nodes[1].op_type == "Identity"

    assert isinstance(import_graph_def({"nodes": [n1]}), LogicalGraph)
    assert isinstance(import_graph_def({}), LogicalGraph)
    assert isinstance(import_graph_def([n1, n2]), LogicalGraph)
    assert isinstance(import_graph_def("unrecognized"), LogicalGraph)

    with pytest.raises(ValueError, match="op_dict"):
        import_graph_def([n1], op_dict={"AllowedOnly": "AllowedOnly"})
    with pytest.raises(ValueError, match="not found"):
        import_graph_def([n1], return_elements=["missing_node"])

    with init_scope():
        pass

    class SymbolicTensor:
        """Test helper class `SymbolicTensor`."""

        is_symbolic = True

    assert is_symbolic_tensor(SymbolicTensor()) is True
    assert is_symbolic_tensor(tensor) is False
    assert is_tensor(tensor) is True
    assert is_tensor(42) is False

    out_idx, out_vals, empty_ind, rev_map = ragged_fill_empty_rows(
        indices=tf.Tensor([[0, 0], [2, 0], [2, 1]], dtype=tf.int64),
        values=tf.Tensor([10.0, 20.0, 30.0]),
        dense_shape=tf.Tensor([4, 2], dtype=tf.int64),
        default_value=-1.0,
    )
    tf.debugging.assert_equal(out_idx, [[0, 0], [1, 0], [2, 0], [2, 1], [3, 0]])
    tf.debugging.assert_near(out_vals, [10.0, -1.0, 20.0, 30.0, -1.0])
    tf.debugging.assert_equal(empty_ind, [False, True, False, True])
    tf.debugging.assert_equal(rev_map, [0, 2, 3])

    out_idx2, _, _, _ = ragged_fill_empty_rows(
        indices=[[1, 0]],
        values=[5.0],
        dense_shape=None,
        default_value=[0.0],
    )
    assert out_idx2.shape == (2, 2)
    out_idx3, _, _, _ = ragged_fill_empty_rows(
        indices=[[5, 0], [0]],
        values=[["nested"]],
        dense_shape=[1, 1],
    )
    assert out_idx3.shape[1] == 2

    d_vals, d_def = ragged_fill_empty_rows_grad(
        reverse_index_map=rev_map,
        grad_values=tf.Tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    tf.debugging.assert_near(d_vals, [1.0, 3.0, 4.0])
    tf.debugging.assert_near(d_def, 7.0)
    d_vals_s, _ = ragged_fill_empty_rows_grad(0, 2.5)
    tf.debugging.assert_near(d_vals_s, [2.5])
    d_vals_empty, _ = ragged_fill_empty_rows_grad("bad", "bad")
    assert d_vals_empty.shape == (0,)

    assert size(tensor) == 2
    assert size(42) == 0

    s2b_in = tf.Tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    s2b_out = space_to_batch_nd(s2b_in, [2, 2], [[0, 0], [0, 0]])
    assert s2b_out is not None and s2b_out.shape == (4, 1, 1, 1)
    tf.debugging.assert_near(
        tf.reshape(s2b_out, (4,)),
        [1.0, 2.0, 3.0, 4.0],
    )

    s2b_pad = space_to_batch(s2b_in, [[2, 0], [0, 2]], block_size=2)
    assert s2b_pad is not None and s2b_pad.shape == (4, 2, 2, 1)
    assert space_to_batch(s2b_in) is not None
    assert space_to_batch(None) is None
    assert space_to_batch_nd(None) is None
    assert space_to_batch_nd(s2b_in, None) is not None
    assert space_to_batch_nd(tf.Tensor([1.0, 2.0]), [2, 2], [[1, 1]]) is not None
    assert space_to_batch_nd(s2b_in, 2, "invalid_pads") is not None

    prev_eager = _compiler_config.eager_mode
    prev_tracing = _tracer.is_tracing
    prev_graph = _tracer.active_graph
    _compiler_config.eager_mode = False
    _tracer.is_tracing = True
    _tracer.active_graph = LogicalGraph(name="s2b_trace")
    try:
        pt = ProxyTensor(id="x_in", shape=(1, 2, 2, 1), dtype="float32")
        traced_res = space_to_batch_nd(tf._to_tensor(pt), [2, 2], None)
        assert isinstance(traced_res, tf.Tensor)
    finally:
        _compiler_config.eager_mode = prev_eager
        _tracer.is_tracing = prev_tracing
        _tracer.active_graph = prev_graph

    assert _to_py_data(_FakeArrayLike()) == [1, 2]
    assert _to_py_data(None) is None
    assert _to_float("not_a_float") == 0.0
    assert _to_int("not_an_int") == 0
    assert string(tensor) is not None
    assert string() is None
