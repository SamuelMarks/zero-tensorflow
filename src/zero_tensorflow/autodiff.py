"""Automatic differentiation (`GradientTape`, `function`, `hessians`, `custom_gradient`, `stop_gradient`)."""

from __future__ import annotations

import builtins
import functools
import itertools
import operator
import uuid
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Generic, TypeVar

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    CallableOperandProtocol,
    TensorOperand,
    TensorSpecProtocol,
    as_operand,
)
from zero_tensorflow.core_tensor import (
    Tensor,
    Variable,
    _extract_constant_value,
    _to_tensor,
    _wrap_tensor,
    to_dtype,
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

_TAPE_STACK: list[GradientTape] = []


class _CustomGradRecord:
    """Record storing a `@custom_gradient` forward output node and its backward callback."""

    def __init__(
        self,
        out_node_id: str,
        out_tensor: Tensor,
        in_tensors: list[Tensor],
        in_node_ids: list[str],
        grad_fn: CallableOperandProtocol,
    ) -> None:
        """Initialize `_CustomGradRecord`.

        Args:
            out_node_id: Node ID of the primal output in the active `LogicalGraph`.
            out_tensor: Primal output `Tensor`.
            in_tensors: Input `Tensor` operands passed to the custom-gradient function.
            in_node_ids: Node IDs of `in_tensors` in the active `LogicalGraph`.
            grad_fn: Backward gradient callback returned by the custom-gradient function.
        """
        self.out_node_id = out_node_id
        self.out_tensor = out_tensor
        self.in_tensors = in_tensors
        self.in_node_ids = in_node_ids
        self.grad_fn = grad_fn


_CUSTOM_GRAD_RECORDS: dict[int, list[_CustomGradRecord]] = {}


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


def _signature_key_for_arg(
    arg: TensorOperand | TensorSpecProtocol,
    reduce_retracing: builtins.bool,
) -> str:
    """Compute a deterministic cache key segment for a single argument or `TensorSpec`.

    Args:
        arg: Input argument or `TensorSpecProtocol`.
        reduce_retracing: Whether to generalize concrete dimension sizes to rank-only.

    Returns:
        str: Cache key segment string.
    """
    if isinstance(arg, Tensor):
        shape_key = (None,) * len(arg.shape) if reduce_retracing else arg.shape
        return f"Tensor({shape_key},{to_dtype(arg.dtype)})"
    if isinstance(arg, TensorSpecProtocol):
        raw_shape = tuple(arg.shape)
        shape_key_spec = (None,) * len(raw_shape) if reduce_retracing else raw_shape
        return f"Tensor({shape_key_spec},{to_dtype(arg.dtype)})"
    return f"{type(arg).__name__}:{arg!r}"


def _build_cache_key(
    args: Sequence[TensorOperand | TensorSpecProtocol],
    kwargs: dict[str, TensorOperand | TensorSpecProtocol],
    reduce_retracing: builtins.bool,
) -> str:
    """Build a canonical cache key for concrete function lookup.

    Args:
        args: Positional arguments or `input_signature` specs.
        kwargs: Keyword arguments.
        reduce_retracing: Whether to generalize shapes to rank-only.

    Returns:
        str: Canonical signature key string.
    """
    arg_parts = [_signature_key_for_arg(a, reduce_retracing) for a in args]
    kw_parts = [
        f"{k}={_signature_key_for_arg(v, reduce_retracing)}" for k, v in sorted(kwargs.items())
    ]
    return "(" + ",".join(arg_parts + kw_parts) + ")"


class _ConcreteFunction(Generic[_P, _R]):
    """Compiled concrete function associated with a fixed input signature."""

    def __init__(
        self,
        func: Callable[_P, _R],
        signature_key: str,
        graph: LogicalGraph,
    ) -> None:
        """Initialize `_ConcreteFunction`.

        Args:
            func: Underlying Python callable.
            signature_key: Canonical cache key string for this specialization.
            graph: Traced `LogicalGraph` instance.
        """
        self._func = func
        self.signature_key = signature_key
        self.graph = graph

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Execute the compiled function for `*args` and `**kwargs`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            _R: Function output.
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
            res: _R = self._func(*args, **kwargs)
        finally:
            _tracer.is_tracing = prev_is_tracing
            _tracer.active_graph = prev_graph

        self.graph = traced_graph
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


class _PolymorphicFunction(Generic[_P, _R]):
    """Polymorphic `@tf.function` wrapper caching `_ConcreteFunction` instances by signature."""

    def __init__(
        self,
        func: Callable[_P, _R],
        input_signature: Sequence[TensorSpecProtocol] | None = None,
        autograph: builtins.bool = True,
        jit_compile: builtins.bool | None = None,
        reduce_retracing: builtins.bool = False,
    ) -> None:
        """Initialize `_PolymorphicFunction`.

        Args:
            func: Python function to compile/trace.
            input_signature: Optional sequence of `TensorSpec` objects constraining inputs.
            autograph: Whether AutoGraph is enabled.
            jit_compile: Optional XLA/JIT compilation flag.
            reduce_retracing: Whether to relax concrete shapes when caching traces.
        """
        self._func = func
        self.input_signature = (
            builtins.tuple(input_signature) if input_signature is not None else None
        )
        self.autograph = autograph
        self.jit_compile = jit_compile
        self.reduce_retracing = reduce_retracing
        self._concrete_cache: dict[str, _ConcreteFunction[_P, _R]] = {}
        functools.update_wrapper(self, func)

    def _validate_input_signature(self, args: Sequence[TensorOperand]) -> None:
        """Validate positional `args` against `self.input_signature` if configured.

        Args:
            args: Positional call arguments.

        Raises:
            ValueError: If argument count or tensor shape/dtype does not match `input_signature`.
        """
        if self.input_signature is None:
            return
        if len(args) != len(self.input_signature):
            raise ValueError(
                f"Expected {len(self.input_signature)} arguments for input_signature, "
                f"got {len(args)}."
            )
        for arg, spec in zip(args, self.input_signature):
            spec_shape = builtins.tuple(spec.shape)
            if isinstance(arg, Tensor) and (
                len(arg.shape) != len(spec_shape)
                or any(sd is not None and sd != ad for sd, ad in zip(spec_shape, arg.shape))
            ):
                raise ValueError(
                    f"Incompatible argument shape {arg.shape} for spec shape {spec_shape}."
                )

    def get_concrete_function(
        self,
        *args: TensorOperand | TensorSpecProtocol,
        **kwargs: TensorOperand | TensorSpecProtocol,
    ) -> _ConcreteFunction[_P, _R]:
        """Return a cached or newly traced `_ConcreteFunction` for the given signature.

        Args:
            *args: Example `Tensor`s or `TensorSpec`s.
            **kwargs: Example keyword `Tensor`s or `TensorSpec`s.

        Returns:
            _ConcreteFunction[_P, _R]: Cached concrete function specialization.
        """
        eff_args: Sequence[TensorOperand | TensorSpecProtocol] = (
            args if args or self.input_signature is None else self.input_signature
        )
        key = _build_cache_key(eff_args, kwargs, self.reduce_retracing)
        if key not in self._concrete_cache:
            self._concrete_cache[key] = _ConcreteFunction(
                self._func,
                key,
                LogicalGraph(name="tf_function"),
            )
        return self._concrete_cache[key]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Validate signature, look up concrete function, and execute.

        Args:
            *args: Positional arguments forwarded to `self._func`.
            **kwargs: Keyword arguments forwarded to `self._func`.

        Returns:
            _R: Function output.
        """
        typed_args: list[TensorOperand] = [
            a if isinstance(a, (Tensor, int, float, bool, str)) else None for a in args
        ]
        self._validate_input_signature(typed_args)
        typed_kwargs: dict[str, TensorOperand | TensorSpecProtocol] = {
            str(k): (v if isinstance(v, (Tensor, int, float, bool, str)) else None)
            for k, v in kwargs.items()
        }
        concrete = self.get_concrete_function(*typed_args, **typed_kwargs)
        return concrete(*args, **kwargs)


def function(
    func: Callable[_P, _R] | None = None,
    input_signature: Sequence[TensorSpecProtocol] | None = None,
    autograph: builtins.bool = True,
    jit_compile: builtins.bool | None = None,
    reduce_retracing: builtins.bool = False,
    experimental_implements: str | None = None,
    experimental_autograph_options: TensorOperand = None,
    experimental_attributes: Mapping[str, TensorOperand] | None = None,
    experimental_relax_shapes: builtins.bool = False,
    experimental_compile: builtins.bool | None = None,
    experimental_follow_type_hints: builtins.bool | None = None,
) -> _PolymorphicFunction[_P, _R]:
    """Compile a Python function into a callable TensorFlow graph.

    Args:
        func: Python function to compile/trace.
        input_signature: Optional sequence of `TensorSpec` instances specifying input types.
        autograph: Whether to enable AutoGraph control-flow conversion.
        jit_compile: Optional boolean enabling JIT compilation.
        reduce_retracing: Whether to trace using generalized shapes to reduce retracing.
        experimental_implements: Optional interface name implemented by this function.
        experimental_autograph_options: Optional tuple of `tf.autograph.experimental.Feature` values.
        experimental_attributes: Optional dictionary of extra function attributes.
        experimental_relax_shapes: Whether to relax argument shapes to avoid retracing.
        experimental_compile: Deprecated alias for `jit_compile`.
        experimental_follow_type_hints: Whether to use type annotations to optimize tracing.

    Returns:
        _PolymorphicFunction[_P, _R]: Polymorphic graph-traced callable.
    """
    del (
        experimental_implements,
        experimental_autograph_options,
        experimental_attributes,
        experimental_relax_shapes,
        experimental_compile,
        experimental_follow_type_hints,
    )
    return _PolymorphicFunction(
        func,  # type: ignore[arg-type]
        input_signature=input_signature,
        autograph=autograph,
        jit_compile=jit_compile,
        reduce_retracing=reduce_retracing,
    )


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


def _ones_like_source(source: Tensor) -> Tensor:
    """Create a ones `Tensor` matching `source`'s shape and dtype.

    Args:
        source: Reference `Tensor` or `Variable`.

    Returns:
        Tensor: Ones tensor with matching shape and dtype.
    """
    prev_tracing = _tracer.is_tracing
    _tracer.is_tracing = False
    try:
        return _wrap_tensor(_ops.ones(source.shape, dtype=source.dtype))
    finally:
        _tracer.is_tracing = prev_tracing


def _materialize_evaluated_grad(
    val: EvalValue | TensorOperand,
    source: Tensor,
) -> Tensor:
    """Convert an evaluated gradient result into a `Tensor` matching `source.dtype`.

    Args:
        val: Raw value produced by `evaluate_graph`.
        source: Source `Tensor` with respect to which the gradient was computed.

    Returns:
        Tensor: Materialized gradient `Tensor`.
    """
    prev_tracing = _tracer.is_tracing
    _tracer.is_tracing = False
    try:
        return _wrap_tensor(_to_tensor(as_operand(val), dtype=source.dtype))
    finally:
        _tracer.is_tracing = prev_tracing


def _add_cotangent_node(
    graph: LogicalGraph,
    cot_tensor: Tensor,
) -> str:
    """Insert a `Constant` cotangent node for `cot_tensor` into `graph`.

    Args:
        graph: Target `LogicalGraph`.
        cot_tensor: Cotangent `Tensor` value.

    Returns:
        str: Inserted `LogicalNode` ID.
    """
    cot_id = f"cotangent_{uuid.uuid4().hex[:6]}"
    raw_data = cot_tensor._tensor.data if cot_tensor._tensor is not None else 1.0
    cot_val = _extract_constant_value(raw_data)
    cot_node = LogicalNode(
        id=cot_id,
        op_type="Constant",
        attributes={"value": cot_val},
        shape_metadata=cot_tensor.shape,
    )
    graph.nodes[cot_id] = cot_node
    return cot_id


def _compute_raw_symbolic_grad(
    graph: LogicalGraph,
    target_id: str,
    wrt_id: str,
    source: Tensor,
    cotangent_id: str | None = None,
    unconnected_gradients: str = "none",
) -> tuple[Tensor | None, LogicalGraph, str | None]:
    """Differentiate `target_id` wrt `wrt_id` on `graph` using symbolic VJP rules.

    Args:
        graph: Traced forward `LogicalGraph`.
        target_id: Output node ID to differentiate.
        wrt_id: Source node ID with respect to which differentiation is performed.
        source: Source `Tensor` or `Variable` instance.
        cotangent_id: Optional cotangent node ID for `target_id`.
        unconnected_gradients: `'none'` or `'zero'` behavior for disconnected nodes.

    Returns:
        tuple[Tensor | None, LogicalGraph, str | None]: `(grad_tensor, grad_graph, grad_out_id)`.
    """
    mode = unconnected_gradients.lower()
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


def _compute_single_grad(
    graph: LogicalGraph,
    target_id: str,
    wrt_id: str,
    source: Tensor,
    cotangent_id: str | None = None,
    unconnected_gradients: str = "none",
) -> tuple[Tensor | None, LogicalGraph, str | None]:
    """Differentiate `target_id` with respect to `wrt_id` on `graph` including `@custom_gradient`s.

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

    records = _CUSTOM_GRAD_RECORDS.get(id(graph), [])
    custom_parts: list[Tensor] = []
    for rec in records:
        if target_id == rec.out_node_id:
            if cotangent_id is not None and cotangent_id in graph.nodes:
                cot_attr = graph.nodes[cotangent_id].attributes.get("value", 1.0)
                upstream_dy = _materialize_evaluated_grad(
                    as_operand(cot_attr),
                    rec.out_tensor,
                )
            else:
                upstream_dy = _ones_like_source(rec.out_tensor)
        else:
            dy_opt, _, _ = _compute_raw_symbolic_grad(
                graph,
                target_id,
                rec.out_node_id,
                rec.out_tensor,
                cotangent_id=cotangent_id,
                unconnected_gradients="zero",
            )
            assert dy_opt is not None
            upstream_dy = dy_opt

        prev_tracing = _tracer.is_tracing
        _tracer.is_tracing = False
        try:
            raw_cg = rec.grad_fn(upstream_dy)
            cg_items: list[TensorOperand] = (
                builtins.list(raw_cg)
                if isinstance(raw_cg, (builtins.list, builtins.tuple))
                else [raw_cg]
            )
            for inp_t, inp_nid, cg_val in zip(
                rec.in_tensors,
                rec.in_node_ids,
                cg_items,
            ):
                cg_tensor = _materialize_evaluated_grad(as_operand(cg_val), inp_t)
                if inp_nid == wrt_id:
                    custom_parts.append(cg_tensor)
                else:
                    sub_cot_id = _add_cotangent_node(graph, cg_tensor)
                    contrib, _, _ = _compute_raw_symbolic_grad(
                        graph,
                        inp_nid,
                        wrt_id,
                        source,
                        cotangent_id=sub_cot_id,
                        unconnected_gradients="zero",
                    )
                    assert contrib is not None
                    custom_parts.append(contrib)
        finally:
            _tracer.is_tracing = prev_tracing

    prev_tracing = _tracer.is_tracing
    _tracer.is_tracing = False
    saved_ops: list[tuple[LogicalNode, str]] = []
    for rec in records:
        node_obj = graph.nodes[rec.out_node_id]
        saved_ops.append((node_obj, node_obj.op_type))
        node_obj.op_type = "StopGradient"
    try:
        base_grad, grad_graph, grad_out_id = _compute_raw_symbolic_grad(
            graph,
            target_id,
            wrt_id,
            source,
            cotangent_id=cotangent_id,
            unconnected_gradients="none" if custom_parts else mode,
        )
    finally:
        for node_obj, orig_op in saved_ops:
            node_obj.op_type = orig_op
        _tracer.is_tracing = prev_tracing

    if custom_parts:
        prev_tracing = _tracer.is_tracing
        _tracer.is_tracing = False
        try:
            custom_total: Tensor = functools.reduce(operator.add, custom_parts)
            combined = custom_total if base_grad is None else (base_grad + custom_total)
        finally:
            _tracer.is_tracing = prev_tracing
        return (combined, grad_graph, grad_out_id)
    return (base_grad, grad_graph, grad_out_id)


def _iter_shape_indices(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Return all multi-indices for a tensor shape.

    Args:
        shape: Tuple of non-negative dimension sizes.

    Returns:
        list[tuple[int, ...]]: Cartesian product of dimension indices.
    """
    return list(itertools.product(*(range(d) for d in shape)))


def _unit_basis_tensor(
    shape: tuple[int, ...],
    index: tuple[int, ...],
    dtype: TensorOperand,
) -> Tensor:
    """Create a tensor of `shape` with `1.0` at `index` and `0.0` elsewhere.

    Args:
        shape: Target tensor shape.
        index: Multi-index at which to place `1.0`.
        dtype: Target tensor `DType`.

    Returns:
        Tensor: Unit basis tensor.
    """
    prev_tracing = _tracer.is_tracing
    _tracer.is_tracing = False
    try:
        flat_vals = [1.0 if idx == index else 0.0 for idx in _iter_shape_indices(shape)]
        reshaped = _k_ops.reshape(_to_tensor(flat_vals, dtype=dtype), shape)
        return _wrap_tensor(reshaped)
    finally:
        _tracer.is_tracing = prev_tracing


def _batched_unit_basis_tensor(
    shape: tuple[int, ...],
    inner_index: tuple[int, ...],
    dtype: TensorOperand,
) -> Tensor:
    """Create a batched tensor of `shape` with `1.0` at `[b, *inner_index]` for all `b`.

    Args:
        shape: Full target tensor shape `(batch_size, *inner_shape)`.
        inner_index: Multi-index within `inner_shape`.
        dtype: Target tensor `DType`.

    Returns:
        Tensor: Batched unit basis tensor.
    """
    prev_tracing = _tracer.is_tracing
    _tracer.is_tracing = False
    try:
        batch_size = shape[0]
        inner_vals = [1.0 if idx == inner_index else 0.0 for idx in _iter_shape_indices(shape[1:])]
        flat_vals = inner_vals * batch_size
        reshaped = _k_ops.reshape(_to_tensor(flat_vals, dtype=dtype), shape)
        return _wrap_tensor(reshaped)
    finally:
        _tracer.is_tracing = prev_tracing


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
        """Enter the `GradientTape` recording context and push a local `LogicalGraph` frame.

        Returns:
            GradientTape: Active tape instance (`self`).
        """
        self._prev_tracer_graph = _tracer.active_graph
        self._prev_is_tracing = _tracer.is_tracing

        self._tape = TracerTape()
        self._graph = self._tape.start_tracing("GradientTape")

        parent_graph = _TAPE_STACK[-1]._graph if _TAPE_STACK else self._prev_tracer_graph
        if parent_graph is not None:
            self._graph.nodes.update(parent_graph.nodes)
        for parent_tape in _TAPE_STACK:
            self_gid = id(self._graph)
            watched_tensors = [w for w in parent_tape.watched if isinstance(w, Tensor)]
            for w in watched_tensors:
                w._traced_node_ids[self_gid] = str(_resolve_node_id(w, parent_tape._graph))

        _TAPE_STACK.append(self)
        _tracer.active_graph = self._graph
        _tracer.is_tracing = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the `GradientTape` recording context and pop the tape frame.

        Args:
            exc_type: Exception type if raised, else None.
            exc_val: Exception value if raised, else None.
            exc_tb: Traceback if raised, else None.
        """
        del exc_type, exc_val, exc_tb
        _TAPE_STACK.pop()
        if _TAPE_STACK:
            parent_tape = _TAPE_STACK[-1]
            parent_tape._graph.nodes.update(self._graph.nodes)
            _CUSTOM_GRAD_RECORDS.setdefault(id(parent_tape._graph), []).extend(
                _CUSTOM_GRAD_RECORDS.get(id(self._graph), [])
            )
            _tracer.active_graph = parent_tape._graph
            _tracer.is_tracing = True
        else:
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
            prev_graph = _tracer.active_graph
            prev_tracing = _tracer.is_tracing
            _tracer.active_graph = self._graph
            _tracer.is_tracing = True
            try:
                _to_tensor(tensor)
            finally:
                _tracer.active_graph = prev_graph
                _tracer.is_tracing = prev_tracing
            self_gid = id(self._graph)
            if self_gid in tensor._traced_node_ids:
                nid = tensor._traced_node_ids[self_gid]
                node = self._graph.nodes[nid]
                for active_tape in _TAPE_STACK:
                    active_gid = id(active_tape._graph)
                    tensor._traced_node_ids.setdefault(active_gid, nid)
                    active_tape._graph.nodes.setdefault(nid, node)

    def watched_variables(self) -> builtins.tuple[Variable, ...]:
        """Return variables watched by this tape in order of construction/watching.

        Returns:
            tuple[Variable, ...]: Tuple of watched `Variable` instances.
        """
        return builtins.tuple(v for v in self.watched if isinstance(v, Variable))

    def reset(self) -> None:
        """Clear all state and operations recorded in this `GradientTape`."""
        self.watched.clear()
        _CUSTOM_GRAD_RECORDS.pop(id(self._graph), None)
        self._graph = LogicalGraph(name="GradientTape")
        self._used = False
        if _TAPE_STACK and _TAPE_STACK[-1] is self:
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
                cotangent_id = _add_cotangent_node(self._graph, og_tensor)

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
            g_tensor, grad_graph, grad_out_id = _compute_single_grad(
                self._graph,
                target_id,
                wrt_id,
                s,
                cotangent_id=cotangent_id,
                unconnected_gradients=unconnected_gradients,
            )
            if g_tensor is not None and grad_out_id is not None and _TAPE_STACK:
                for active_tape in _TAPE_STACK:
                    active_tape._graph.nodes.update(grad_graph.nodes)
                    g_tensor._traced_node_ids[id(active_tape._graph)] = grad_out_id
            res.append(g_tensor)

        return res[0] if is_single else res

    def jacobian(
        self,
        target: Tensor | TensorOperand,
        sources: Tensor | Sequence[Tensor | TensorOperand] | TensorOperand,
        experimental_use_pfor: builtins.bool = True,
        unconnected_gradients: str = "none",
    ) -> Tensor | list[Tensor | None] | None:
        """Compute the Jacobian of `target` with respect to `sources` (shape `target.shape + source.shape`).

        Args:
            target: Tensor to be differentiated.
            sources: A Tensor or sequence of Tensors or Variables.
            experimental_use_pfor: Whether to use vectorized pfor evaluation.
            unconnected_gradients: How to handle unconnected gradients (`'none'` or `'zero'`).

        Returns:
            Tensor | list[Tensor | None] | None: Jacobian tensor(s) of shape `target.shape + source.shape`.
        """
        if not isinstance(target, Tensor):
            return None
        if not target.shape:
            return self.gradient(
                target,
                sources,
                unconnected_gradients=unconnected_gradients,
            )

        was_persistent = self.persistent
        self.persistent = True
        try:
            is_single = not isinstance(sources, (builtins.list, builtins.tuple))
            sources_list: list[Tensor | TensorOperand] = (
                builtins.list(sources)
                if isinstance(sources, (builtins.list, builtins.tuple))
                else [sources]
            )
            target_indices = _iter_shape_indices(target.shape)
            basis_tensors = [
                _unit_basis_tensor(target.shape, idx, target.dtype) for idx in target_indices
            ]
            if not experimental_use_pfor:
                basis_tensors = list(basis_tensors)

            jac_out: list[Tensor | None] = []
            for s in sources_list:
                if not isinstance(s, Tensor):
                    jac_out.append(None)
                    continue
                slice_grads: list[Tensor | None] = []
                for basis in basis_tensors:
                    g_slice = self.gradient(
                        target,
                        s,
                        output_gradients=basis,
                        unconnected_gradients=unconnected_gradients,
                    )
                    slice_grads.append(g_slice if isinstance(g_slice, Tensor) else None)

                if all(g is None for g in slice_grads):
                    jac_out.append(None)
                else:
                    prev_tracing = _tracer.is_tracing
                    _tracer.is_tracing = False
                    try:
                        filled = [
                            _to_tensor(g if g is not None else _zero_like_source(s))
                            for g in slice_grads
                        ]
                        stacked = _k_ops.stack(filled, axis=0)
                        reshaped = _k_ops.reshape(stacked, target.shape + s.shape)
                        jac_out.append(_wrap_tensor(reshaped))
                    finally:
                        _tracer.is_tracing = prev_tracing
        finally:
            self.persistent = was_persistent
            if not was_persistent:
                self._used = True

        return jac_out[0] if is_single else jac_out

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
            Tensor | list[Tensor | None] | None: Batch Jacobian tensor(s) of shape `[b, y1, ..., y_n, x1, ..., x_m]`.
        """
        if not isinstance(target, Tensor):
            return None
        is_single = not isinstance(sources, (builtins.list, builtins.tuple))
        sources_list: list[Tensor | TensorOperand] = (
            builtins.list(sources)
            if isinstance(sources, (builtins.list, builtins.tuple))
            else [sources]
        )

        was_persistent = self.persistent
        self.persistent = True
        try:
            batch_out: list[Tensor | None] = []
            for s_val in sources_list:
                if not isinstance(s_val, Tensor):
                    batch_out.append(None)
                    continue
                if len(target.shape) < 2 or len(s_val.shape) < 2:
                    g_scalar = self.gradient(
                        target,
                        s_val,
                        unconnected_gradients=unconnected_gradients,
                    )
                    batch_out.append(g_scalar if isinstance(g_scalar, Tensor) else None)
                    continue

                batch_size = target.shape[0]
                target_inner = target.shape[1:]
                source_inner = s_val.shape[1:]
                inner_indices = _iter_shape_indices(target_inner)
                basis_list = [
                    _batched_unit_basis_tensor(target.shape, idx, target.dtype)
                    for idx in inner_indices
                ]
                if not experimental_use_pfor:
                    basis_list = list(basis_list)

                slice_grads: list[Tensor | None] = []
                for basis in basis_list:
                    g_slice = self.gradient(
                        target,
                        s_val,
                        output_gradients=basis,
                        unconnected_gradients=unconnected_gradients,
                    )
                    slice_grads.append(g_slice if isinstance(g_slice, Tensor) else None)

                if all(g is None for g in slice_grads):
                    batch_out.append(None)
                else:
                    prev_tracing = _tracer.is_tracing
                    _tracer.is_tracing = False
                    try:
                        filled = [
                            _to_tensor(g if g is not None else _zero_like_source(s_val))
                            for g in slice_grads
                        ]
                        stacked = _k_ops.stack(filled, axis=1)
                        out_shape = (batch_size, *target_inner, *source_inner)
                        reshaped = _k_ops.reshape(stacked, out_shape)
                        batch_out.append(_wrap_tensor(reshaped))
                    finally:
                        _tracer.is_tracing = prev_tracing
        finally:
            self.persistent = was_persistent
            if not was_persistent:
                self._used = True

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
        """Wrap `fn` so calling it returns the primal output `Tensor` and registers `grad_fn`.

        Args:
            fn: Target custom-gradient function.

        Returns:
            Callable[_P, Tensor]: Wrapped primal function.
        """

        @functools.wraps(fn)
        def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> Tensor:
            """Invoke `fn`, extract the forward output tensor, and record `grad_fn` on active tapes.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.

            Returns:
                Tensor: Primal forward output tensor.
            """
            was_tracing = _tracer.is_tracing
            active_graph = _tracer.active_graph
            in_tensors: list[Tensor] = []
            in_node_ids: list[str] = []

            if was_tracing and active_graph is not None:
                tensor_args = [a for a in args if isinstance(a, Tensor)]
                for a in tensor_args:
                    _to_tensor(a)
                    in_tensors.append(a)
                    in_node_ids.append(str(_resolve_node_id(a, active_graph)))

            out = fn(*args, **kwargs)
            if isinstance(out, builtins.tuple):
                y_tensor = _wrap_tensor(out[0])
                grad_fn = out[1]
            else:
                y_tensor = _wrap_tensor(out)
                grad_fn = None

            if was_tracing and active_graph is not None and grad_fn is not None and in_tensors:
                _to_tensor(y_tensor)
                out_nid = str(_resolve_node_id(y_tensor, active_graph))
                record = _CustomGradRecord(
                    out_node_id=out_nid,
                    out_tensor=y_tensor,
                    in_tensors=in_tensors,
                    in_node_ids=in_node_ids,
                    grad_fn=grad_fn,
                )
                _CUSTOM_GRAD_RECORDS.setdefault(id(active_graph), []).append(record)
            return y_tensor

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


def _zero_hessian_for_source(source: Tensor) -> Tensor:
    """Create a zero Hessian `Tensor` of shape `source.shape + source.shape`.

    Args:
        source: Differentiation source `Tensor`.

    Returns:
        Tensor: Zero tensor with shape `source.shape + source.shape`.
    """
    prev_tracing = _tracer.is_tracing
    _tracer.is_tracing = False
    try:
        return _wrap_tensor(_ops.zeros(source.shape + source.shape, dtype=source.dtype))
    finally:
        _tracer.is_tracing = prev_tracing


def hessians(
    ys: Tensor | Sequence[Tensor] | TensorOperand,
    xs: Tensor | Sequence[Tensor] | TensorOperand,
    gate_gradients: builtins.bool = False,
    aggregation_method: int | None = None,
    name: str = "hessians",
) -> list[Tensor]:
    """Construct the Hessian of sum of `ys` with respect to `xs` (shape `x.shape + x.shape`).

    Args:
        ys: A `Tensor` or list of `Tensor`s to be differentiated.
        xs: A `Tensor` or list of `Tensor`s to be used for differentiation.
        gate_gradients: Whether to add a tuple around the gradients returned for an op.
        aggregation_method: Optional aggregation method.
        name: Optional name prefix for the returned tensors.

    Returns:
        list[Tensor]: List of Hessian `Tensor`s of shape `x.shape + x.shape` for each `x` in `xs`.
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
                if not x.shape:
                    h_tensor, _, _ = _compute_single_grad(
                        g1_graph,
                        str(g1_out),
                        x_id,
                        x,
                        unconnected_gradients="zero",
                    )
                    computed = h_tensor
                else:
                    row_slices: list[Tensor] = []
                    for idx in _iter_shape_indices(x.shape):
                        basis = _unit_basis_tensor(x.shape, idx, x.dtype)
                        cot_id = _add_cotangent_node(g1_graph, basis)
                        h_row, _, _ = _compute_single_grad(
                            g1_graph,
                            str(g1_out),
                            x_id,
                            x,
                            cotangent_id=cot_id,
                            unconnected_gradients="zero",
                        )
                        row_slices.append(h_row if h_row is not None else _zero_like_source(x))
                    prev_tracing = _tracer.is_tracing
                    _tracer.is_tracing = False
                    try:
                        stacked = _k_ops.stack([_to_tensor(r) for r in row_slices], axis=0)
                        reshaped = _k_ops.reshape(stacked, x.shape + x.shape)
                        computed = _wrap_tensor(reshaped)
                    finally:
                        _tracer.is_tracing = prev_tracing

        results.append(computed if computed is not None else _zero_hessian_for_source(x))

    return results


__all__ = [
    "GradientTape",
    "custom_gradient",
    "function",
    "hessians",
    "stop_gradient",
]
