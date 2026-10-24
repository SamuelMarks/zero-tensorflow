"""Unit tests for `test_ops_control_flow`."""

from __future__ import annotations

import pytest
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import control_flow
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_ir import LogicalGraph


def test_control_flow_tracing() -> None:
    """Verify control flow ops (`cond`, `while_loop`, `scan`, `vmap`, `pmap`) under tracing."""
    prev_tracing = _tracer.is_tracing
    prev_graph = _tracer.active_graph
    proxy_x = ProxyTensor("x_id", (2, 2), DType.Float32.value)
    x = Tensor(proxy_x, TensorConfig(shape=(2, 2), dtype=DType.Float32, device="cpu"))
    try:
        _tracer.is_tracing = True
        _tracer.active_graph = LogicalGraph(name="cf_test")

        def true_fn() -> Tensor:
            """Return `x`.

            Returns:
                Tensor: Tensor `x`.
            """
            return x

        def false_fn() -> Tensor:
            """Return `x`.

            Returns:
                Tensor: Tensor `x`.
            """
            return x

        control_flow.cond(x, true_fn, false_fn)

        def cond_fn(val: Tensor) -> Tensor:
            """Return condition tensor `x`.

            Args:
                val: Loop state value.

            Returns:
                Tensor: Condition tensor.
            """
            del val
            return x

        def body_fn(val: Tensor) -> Tensor:
            """Return loop body state `val`.

            Args:
                val: Loop state value.

            Returns:
                Tensor: Updated loop state.
            """
            return val

        control_flow.while_loop(cond_fn, body_fn, x)

        def scan_fn(c: Tensor, xs: Tensor) -> tuple[Tensor, Tensor]:
            """Return `(c, xs)` scan step tuple.

            Args:
                c: Carry state.
                xs: Input slice.

            Returns:
                tuple[Tensor, Tensor]: Updated `(carry, output)`.
            """
            return (c, xs)

        control_flow.scan(scan_fn, x, x)

        def vmap_fn(val: Tensor) -> Tensor:
            """Return `val` unchanged.

            Args:
                val: Input value.

            Returns:
                Tensor: Output value.
            """
            return val

        vmapped = control_flow.vmap(vmap_fn)
        assert callable(vmapped)
        vmapped(x)
        pmapped = control_flow.pmap(vmap_fn)
        assert callable(pmapped)
        pmapped(x)
        control_flow.stop_gradient(x)
        control_flow.stop_gradient(proxy_x)
        _tracer.is_tracing = False
        prev_eager = config.eager_mode
        config.eager_mode = False
        try:
            with pytest.raises((RuntimeError, ValueError, TypeError)):
                control_flow.cond(x, true_fn, false_fn)
            with pytest.raises((RuntimeError, ValueError, TypeError)):
                control_flow.while_loop(cond_fn, body_fn, x)
            with pytest.raises((RuntimeError, ValueError, TypeError)):
                control_flow.scan(scan_fn, x, x)
            with pytest.raises((RuntimeError, ValueError, TypeError)):
                vmapped(x)
            with pytest.raises((RuntimeError, ValueError, TypeError)):
                pmapped(x)
        finally:
            config.eager_mode = prev_eager
    finally:
        _tracer.is_tracing = prev_tracing
        _tracer.active_graph = prev_graph

    def true_fn_tuple() -> tuple[Tensor, Tensor]:
        """Return tuple `(x, x)`.

        Returns:
            tuple[Tensor, Tensor]: Pair of tensors.
        """
        return (x, x)

    def false_fn_tuple() -> tuple[Tensor, Tensor]:
        """Return tuple `(x, x)`.

        Returns:
            tuple[Tensor, Tensor]: Pair of tensors.
        """
        return (x, x)

    try:
        _tracer.is_tracing = True
        _tracer.active_graph = LogicalGraph(name="cf_tuple_test")
        control_flow.cond(x, true_fn_tuple, false_fn_tuple)
    finally:
        _tracer.is_tracing = prev_tracing
        _tracer.active_graph = prev_graph
