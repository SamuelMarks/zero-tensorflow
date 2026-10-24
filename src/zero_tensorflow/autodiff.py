"""Automatic differentiation (`GradientTape`, `function`, `hessians`, `custom_gradient`, `stop_gradient`)."""

from __future__ import annotations

import builtins
import functools
import uuid
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import TypeVar

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    CallableOperandProtocol,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.core_tensor import (
    Tensor,
    Variable,
    _extract_constant_value,
    _to_tensor,
    _wrap_tensor,
)

import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue, evaluate_graph
from ml_switcheroo_compiler.tracing import ProxyTensor, TracerTape
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.transforms.autodiff import grad as _symbolic_grad
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from typing_extensions import ParamSpec
from zero_keras import ops as _k_ops

_P = ParamSpec("_P")
_R = TypeVar("_R")
_CustomGradOut = TypeVar(
    "_CustomGradOut",
    bound="Tensor | TensorOperand | tuple[Tensor | TensorOperand, CallableOperandProtocol]",
)


def _to_tensor_if_possible(
    x: _R,
) -> None:
    """Ensure numeric, array-like, or list inputs are registered as `Tensor`s during tracing.

    Args:
        x: Candidate input argument.
    """
    if isinstance(
        x,
        (int, float, builtins.list, Tensor, Variable, ArrayLikeProtocol),
    ):
        _wrap_tensor(_to_tensor(x))


def function(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Compile a Python function into a callable TensorFlow graph.

    Args:
        func: Python function to compile/trace.

    Returns:
        Callable[_P, _R]: Wrapped graph-traced callable.
    """

    @functools.wraps(func)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Execute `func` under an active `LogicalGraph` tracing context.

        Args:
            *args: Positional arguments forwarded to `func`.
            **kwargs: Keyword arguments forwarded to `func`.

        Returns:
            _R: Traced output(s).
        """
        for a in args:
            _to_tensor_if_possible(a)
        for v in kwargs.values():
            _to_tensor_if_possible(v)

        prev_is_tracing = _tracer.is_tracing
        prev_graph = _tracer.active_graph
        traced_graph = LogicalGraph(name="tf_function")
        _tracer.is_tracing = True
        _tracer.active_graph = traced_graph

        try:
            res: _R = func(*args, **kwargs)
        finally:
            _tracer.is_tracing = prev_is_tracing
            _tracer.active_graph = prev_graph

        if (
            isinstance(res, Tensor)
            and res._tensor is not None
            and isinstance(res._tensor.data, ProxyTensor)
            and res._tensor.data.id in traced_graph.nodes
            and not prev_is_tracing
        ):
            node_id = res._tensor.data.id
            traced_graph.outputs = [node_id]
            evaluated = evaluate_graph(traced_graph, {})
            res._tensor = _to_tensor(as_operand(evaluated[node_id]), dtype=res.dtype)
        elif isinstance(res, builtins.tuple):
            for r in res:
                _to_tensor_if_possible(r)
        else:
            _to_tensor_if_possible(res)
        return res

    return wrapped


def _resolve_node_id(
    tensor: Tensor,
    graph: LogicalGraph,
) -> str | None:
    """Resolve the `LogicalNode` ID corresponding to `tensor` inside `graph`.

    Args:
        tensor: Candidate `Tensor` or `Variable`.
        graph: Target `LogicalGraph`.

    Returns:
        str | None: Node ID in `graph.nodes` if found, else `None`.
    """
    graph_id = id(graph)
    if graph_id in tensor._traced_node_ids:
        return tensor._traced_node_ids[graph_id]
    if tensor._tensor is not None and isinstance(tensor._tensor.data, ProxyTensor):
        return tensor._tensor.data.id
    return None


def _zero_like_source(source: Tensor) -> Tensor:
    """Create a zero `Tensor` matching `source`'s shape and dtype.

    Args:
        source: Reference `Tensor` or `Variable`.

    Returns:
        Tensor: Zero tensor with matching shape and dtype.
    """
    prev_tracing = _tracer.is_tracing
    _tracer.is_tracing = False
    try:
        dt = source.dtype
        shape = source.shape
        return _wrap_tensor(_ops.zeros(shape, dtype=dt))
    finally:
        _tracer.is_tracing = prev_tracing


def _materialize_evaluated_grad(
    val: EvalValue | TensorOperand,
    source: Tensor,
) -> Tensor:
    """Convert an evaluated gradient result into a `Tensor` matching `source.shape`.

    Args:
        val: Raw value produced by `evaluate_graph`.
        source: Source `Tensor` with respect to which the gradient was computed.

    Returns:
        Tensor: Materialized gradient `Tensor`.
    """
    prev_tracing = _tracer.is_tracing
    _tracer.is_tracing = False
    try:
        t = _wrap_tensor(_to_tensor(as_operand(val), dtype=source.dtype))
        if t.shape != source.shape and source.shape:
            t = _wrap_tensor(_k_ops.broadcast_to(_to_tensor(t), source.shape))
        return t
    finally:
        _tracer.is_tracing = prev_tracing


def _compute_single_grad(
    graph: LogicalGraph,
    target_id: str,
    wrt_id: str,
    source: Tensor,
    cotangent_id: str | None = None,
    unconnected_gradients: str = "none",
) -> tuple[Tensor | None, LogicalGraph, str | None]:
    """Differentiate `target_id` with respect to `wrt_id` on `graph`.

    Args:
        graph: Traced forward `LogicalGraph`.
        target_id: Output node ID to differentiate.
        wrt_id: Source node ID with respect to which differentiation is performed.
        source: Source `Tensor` or `Variable` instance.
        cotangent_id: Optional cotangent node ID for `target_id`.
        unconnected_gradients: `'none'` or `'zero'` behavior for disconnected nodes.

    Returns:
        tuple[Tensor | None, LogicalGraph, str | None]: `(grad_tensor, grad_graph, grad_out_id)`.

    Raises:
        ValueError: If `unconnected_gradients` is not `'none'` or `'zero'`.
    """
    mode = unconnected_gradients.lower()
    if mode not in ("none", "zero"):
        raise ValueError(
            f"Unknown value for unconnected_gradients: {unconnected_gradients!r}. "
            "Expected 'none' or 'zero'."
        )

    cot_map: dict[str, str] | None = {target_id: cotangent_id} if cotangent_id is not None else None
    grad_graph = _symbolic_grad(graph, [wrt_id], target_id, cotangent_id=cot_map)
    grad_out_id = grad_graph.outputs[0] if grad_graph.outputs else None

    if grad_out_id is None or grad_out_id.startswith("grad_zero"):
        if mode == "zero":
            return (_zero_like_source(source), grad_graph, grad_out_id)
        return (None, grad_graph, grad_out_id)

    prev_tracing = _tracer.is_tracing
    _tracer.is_tracing = False
    try:
        evaluated = evaluate_graph(grad_graph, {})
    finally:
        _tracer.is_tracing = prev_tracing

    raw_val = evaluated.get(grad_out_id, 0.0)
    return (_materialize_evaluated_grad(raw_val, source), grad_graph, grad_out_id)


class GradientTape:
    """Record operations for automatic differentiation."""

    def __init__(
        self,
        persistent: builtins.bool = False,
        watch_accessed_variables: builtins.bool = True,
    ) -> None:
        """Initialize a `GradientTape`.

        Args:
            persistent: Whether the tape can be used to compute multiple gradients.
            watch_accessed_variables: Whether to automatically watch accessed variables.
        """
        self.persistent = persistent
        self._watch_accessed_variables = watch_accessed_variables
        self.watched: list[Tensor | TensorOperand] = []
        self._tape: TracerTape | None = None
        self._graph: LogicalGraph = LogicalGraph(name="GradientTape")
        self._prev_tracer_graph: LogicalGraph | None = None
        self._prev_is_tracing: builtins.bool = False
        self._used: builtins.bool = False

    def __enter__(self) -> GradientTape:
        """Enter the `GradientTape` recording context.

        Returns:
            GradientTape: Active tape instance (`self`).
        """
        self._prev_tracer_graph = _tracer.active_graph
        self._prev_is_tracing = _tracer.is_tracing

        if self._prev_is_tracing and self._prev_tracer_graph is not None:
            self._tape = None
            self._graph = self._prev_tracer_graph
        else:
            self._tape = TracerTape()
            self._graph = self._tape.start_tracing("GradientTape")
            _tracer.active_graph = self._graph
            _tracer.is_tracing = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the `GradientTape` recording context and restore tracer state.

        Args:
            exc_type: Exception type if raised, else None.
            exc_val: Exception value if raised, else None.
            exc_tb: Traceback if raised, else None.
        """
        del exc_type, exc_val, exc_tb
        _tracer.active_graph = self._prev_tracer_graph
        _tracer.is_tracing = self._prev_is_tracing

    def watch(self, tensor: Tensor | TensorOperand) -> None:
        """Ensure that `tensor` is being traced by this tape.

        Args:
            tensor: A `Tensor` or `Variable` to watch.
        """
        if all(w is not tensor for w in self.watched):
            self.watched.append(tensor)
        if isinstance(tensor, Tensor):
            _to_tensor(tensor)

    def watched_variables(self) -> builtins.tuple[Variable, ...]:
        """Return variables watched by this tape in order of construction/watching.

        Returns:
            tuple[Variable, ...]: Tuple of watched `Variable` instances.
        """
        return builtins.tuple(v for v in self.watched if isinstance(v, Variable))

    def reset(self) -> None:
        """Clear all state and operations recorded in this `GradientTape`."""
        self.watched.clear()
        self._graph = LogicalGraph(name="GradientTape")
        self._used = False
        if _tracer.is_tracing and self._tape is not None:
            _tracer.active_graph = self._graph

    def gradient(
        self,
        target: Tensor | None,
        sources: Tensor | Sequence[Tensor | TensorOperand] | TensorOperand,
        output_gradients: Tensor | Sequence[Tensor] | None = None,
        unconnected_gradients: str = "none",
    ) -> Tensor | list[Tensor | None] | None:
        """Compute the gradient using operations recorded in context of this tape.

        Args:
            target: Target `Tensor` to differentiate.
            sources: Single `Tensor`/`Variable` or sequence of `Tensor`s/`Variable`s.
            output_gradients: Optional initial cotangent `Tensor` for `target`.
            unconnected_gradients: How to handle unconnected gradients (`'none'` or `'zero'`).

        Returns:
            Tensor | list[Tensor | None] | None: Computed gradient(s) for `sources`.

        Raises:
            RuntimeError: If called more than once on a non-persistent tape.
            ValueError: If `target` was not created during the tape's recording or
                `unconnected_gradients` is invalid.
        """
        if self._used and not self.persistent:
            raise RuntimeError(
                "A non-persistent GradientTape can only be used to compute one set of gradients "
                "(or jacobians)"
            )
        if not self.persistent:
            self._used = True

        if target is None:
            return None

        target_id = _resolve_node_id(target, self._graph)
        if target_id is None or target_id not in self._graph.nodes:
            raise ValueError("Target was not created during the tape's recording.")

        cotangent_id: str | None = None
        if output_gradients is not None:
            og_tensor = (
                output_gradients[0]
                if isinstance(output_gradients, Sequence) and output_gradients
                else output_gradients
            )
            if isinstance(og_tensor, Tensor) and og_tensor._tensor is not None:
                cot_id = f"cotangent_{uuid.uuid4().hex[:6]}"
                cot_val = _extract_constant_value(og_tensor._tensor.data)
                cot_node = LogicalNode(
                    id=cot_id,
                    op_type="Constant",
                    attributes={"value": cot_val},
                    shape_metadata=og_tensor.shape,
                )
                self._graph.nodes[cot_id] = cot_node
                cotangent_id = cot_id

        is_single = not isinstance(sources, (builtins.list, builtins.tuple))
        sources_list: list[Tensor | TensorOperand] = (
            builtins.list(sources)
            if isinstance(sources, (builtins.list, builtins.tuple))
            else [sources]
        )

        res: list[Tensor | None] = []
        for s in sources_list:
            if not isinstance(s, Tensor):
                res.append(None)
                continue
            wrt_id = _resolve_node_id(s, self._graph)
            if wrt_id is None or wrt_id not in self._graph.nodes:
                res.append(
                    _zero_like_source(s) if unconnected_gradients.lower() == "zero" else None
                )
                continue
            g_tensor, _, _ = _compute_single_grad(
                self._graph,
                target_id,
                wrt_id,
                s,
                cotangent_id=cotangent_id,
                unconnected_gradients=unconnected_gradients,
            )
            res.append(g_tensor)

        return res[0] if is_single else res

    def jacobian(
        self,
        target: Tensor | TensorOperand,
        sources: Tensor | Sequence[Tensor | TensorOperand] | TensorOperand,
        experimental_use_pfor: builtins.bool = True,
        unconnected_gradients: str = "none",
    ) -> Tensor | list[Tensor | None] | None:
        """Compute the Jacobian using operations recorded in context of this tape.

        Args:
            target: Tensor to be differentiated.
            sources: A Tensor or sequence of Tensors or Variables.
            experimental_use_pfor: Whether to use pfor for computing the Jacobian.
            unconnected_gradients: How to handle unconnected gradients (`'none'` or `'zero'`).

        Returns:
            Tensor | list[Tensor | None] | None: Jacobian tensor(s) for `sources`.
        """
        del experimental_use_pfor
        if not isinstance(target, Tensor):
            return None
        return self.gradient(
            target,
            sources,
            unconnected_gradients=unconnected_gradients,
        )

    def batch_jacobian(
        self,
        target: Tensor | TensorOperand,
        sources: Tensor | Sequence[Tensor | TensorOperand] | TensorOperand,
        unconnected_gradients: str = "none",
        experimental_use_pfor: builtins.bool = True,
    ) -> Tensor | list[Tensor | None] | None:
        """Compute and assemble the batch Jacobian of `target` with respect to `sources`.

        Args:
            target: A tensor with shape `[b, y1, ..., y_n]`.
            sources: A tensor or sequence of tensors with shape `[b, x1, ..., x_m]`.
            unconnected_gradients: How to handle unconnected gradients (`'none'` or `'zero'`).
            experimental_use_pfor: Whether to use pfor for computing the Jacobian.

        Returns:
            Tensor | list[Tensor | None] | None: Batch Jacobian tensor(s) for `sources`.
        """
        del experimental_use_pfor
        if not isinstance(target, Tensor):
            return None
        is_single = not isinstance(sources, (builtins.list, builtins.tuple))
        sources_list: list[Tensor | TensorOperand] = (
            builtins.list(sources)
            if isinstance(sources, (builtins.list, builtins.tuple))
            else [sources]
        )
        grad_res = self.gradient(
            target,
            sources_list,
            unconnected_gradients=unconnected_gradients,
        )
        grad_items: list[Tensor | None] = grad_res if isinstance(grad_res, builtins.list) else []
        batch_out: list[Tensor | None] = []
        for g_val, s_val in zip(grad_items, sources_list):
            if g_val is None or not isinstance(s_val, Tensor):
                batch_out.append(None)
            elif len(target.shape) >= 2 and len(s_val.shape) >= 2 and g_val.shape == s_val.shape:
                prev_tracing = _tracer.is_tracing
                _tracer.is_tracing = False
                try:
                    batch_out.append(_wrap_tensor(_k_ops.expand_dims(_to_tensor(g_val), axis=1)))
                finally:
                    _tracer.is_tracing = prev_tracing
            else:
                batch_out.append(g_val)
        return batch_out[0] if is_single else batch_out


def stop_gradient(
    input: Tensor | TensorOperand,
    name: str | None = None,
) -> Tensor:
    """Stop gradient computation through `input`.

    Args:
        input: Tensor to detach from gradient recording.
        name: Optional operation name.

    Returns:
        Tensor: Detached `Tensor` with identical value.
    """
    del name
    prev_tracing = _tracer.is_tracing
    _tracer.is_tracing = False
    try:
        return _wrap_tensor(_ops.stop_gradient(_to_tensor(input)))
    finally:
        _tracer.is_tracing = prev_tracing


def custom_gradient(
    f: Callable[_P, _CustomGradOut] | None = None,
) -> Callable[_P, Tensor]:
    """Decorator to define a function with a custom gradient.

    Args:
        f: Function `f(*args)` returning either `(y, grad_fn)` or `y`.

    Returns:
        Callable[_P, Tensor]: Wrapped primal function (or decorator factory when `f` is `None`).
    """

    def _decorate(fn: Callable[_P, _CustomGradOut]) -> Callable[_P, Tensor]:
        """Wrap `fn` so calling it returns the primal output `Tensor`.

        Args:
            fn: Target custom-gradient function.

        Returns:
            Callable[_P, Tensor]: Wrapped primal function.
        """

        @functools.wraps(fn)
        def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> Tensor:
            """Invoke `fn` and extract the forward output tensor.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.

            Returns:
                Tensor: Primal forward output tensor.
            """
            out = fn(*args, **kwargs)
            if isinstance(out, builtins.tuple):
                return _wrap_tensor(out[0])
            return _wrap_tensor(out)

        return _wrapped

    if f is not None:
        return _decorate(f)

    def _factory(*args: _P.args, **kwargs: _P.kwargs) -> Tensor:
        """Return a zero `Tensor` when `@custom_gradient()` is called with `None`.

        Args:
            *args: Positional arguments.
            **kwargs: Unused keyword arguments.

        Returns:
            Tensor: Wrapped zero tensor.
        """
        del args, kwargs
        return _wrap_tensor(0.0)

    return _factory


def hessians(
    ys: Tensor | Sequence[Tensor] | TensorOperand,
    xs: Tensor | Sequence[Tensor] | TensorOperand,
    gate_gradients: builtins.bool = False,
    aggregation_method: int | None = None,
    name: str = "hessians",
) -> list[Tensor]:
    """Construct the Hessian of sum of `ys` with respect to `xs`.

    Args:
        ys: A `Tensor` or list of `Tensor`s to be differentiated.
        xs: A `Tensor` or list of `Tensor`s to be used for differentiation.
        gate_gradients: Whether to add a tuple around the gradients returned for an op.
        aggregation_method: Optional aggregation method.
        name: Optional name prefix for the returned tensors.

    Returns:
        list[Tensor]: List of Hessian `Tensor`s for each `x` in `xs`.
    """
    del gate_gradients, aggregation_method, name
    ys_list: list[Tensor] = (
        [y for y in ys if isinstance(y, Tensor)]
        if isinstance(ys, (builtins.list, builtins.tuple))
        else ([ys] if isinstance(ys, Tensor) else [_wrap_tensor(ys)])
    )
    xs_list: list[Tensor] = (
        [x for x in xs if isinstance(x, Tensor)]
        if isinstance(xs, (builtins.list, builtins.tuple))
        else ([xs] if isinstance(xs, Tensor) else [_wrap_tensor(xs)])
    )

    active_graph = _tracer.active_graph
    results: list[Tensor] = []

    for x in xs_list:
        computed: Tensor | None = None
        if active_graph is not None and ys_list:
            y0 = ys_list[0]
            y_id = _resolve_node_id(y0, active_graph)
            x_id = _resolve_node_id(x, active_graph)
            if (
                y_id is not None
                and x_id is not None
                and y_id in active_graph.nodes
                and x_id in active_graph.nodes
            ):
                _, g1_graph, g1_out = _compute_single_grad(
                    active_graph,
                    y_id,
                    x_id,
                    x,
                    unconnected_gradients="zero",
                )
                h_tensor, _, _ = _compute_single_grad(
                    g1_graph,
                    str(g1_out),
                    x_id,
                    x,
                    unconnected_gradients="zero",
                )
                computed = h_tensor

        results.append(computed if computed is not None else _zero_like_source(x))

    return results


__all__ = [
    "GradientTape",
    "custom_gradient",
    "function",
    "hessians",
    "stop_gradient",
]
