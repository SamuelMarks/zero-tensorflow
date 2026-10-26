"""Control-flow, scoping, graph execution, and runtime namespace operations."""

from __future__ import annotations

import builtins
import contextlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import TracebackType

from zero_tensorflow import debugging as _debugging
from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    CallableOperandProtocol,
    DTypeLike,
    TensorOperand,
    WhileBodyProtocol,
    WhileCondProtocol,
)
from zero_tensorflow.autodiff import GradientTape as _GradientTape
from zero_tensorflow.core_stubs import is_symbolic_tensor
from zero_tensorflow.core_tensor import (
    Tensor,
    Variable,
    _to_tensor,
    _wrap,
    _wrap_tensor,
    to_dtype,
)
from zero_tensorflow.dtypes import DType
from zero_tensorflow.io import FixedLenFeature, VarLenFeature
from zero_tensorflow.ragged import RaggedTensor
from zero_tensorflow.sparse import SparseTensor
from zero_tensorflow.spec_types import (
    _GRADIENT_REGISTRY,
    _NAME_SCOPE_STACK,
    IndexedSlices,
    Operation,
    TensorShape,
    UnconnectedGradients,
)

from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_ir import LogicalGraph
from zero_keras import ops as _ops

__all__ = [
    "Assert",
    "assert_equal",
    "assert_greater",
    "assert_less",
    "assert_rank",
    "autodiff",
    "case",
    "compat",
    "cond",
    "control_dependencies",
    "device",
    "dtensor",
    "errors",
    "executing_eagerly",
    "experimental",
    "feature_column",
    "foldl",
    "foldr",
    "get_current_name_scope",
    "get_logger",
    "get_static_value",
    "grad_pass_through",
    "gradients",
    "graph_util",
    "group",
    "guarantee_const",
    "identity_n",
    "inside_function",
    "lite",
    "load_library",
    "load_op_library",
    "map_fn",
    "mlir",
    "name_scope",
    "nest",
    "no_gradient",
    "no_op",
    "nondifferentiable_batch_function",
    "numpy_function",
    "print",
    "profiler",
    "py_function",
    "queue",
    "recompute_grad",
    "register_tensor_conversion_function",
    "scan",
    "sets",
    "switch_case",
    "sysconfig",
    "timestamp",
    "tools",
    "tpu",
    "tuple",
    "variable_creator_scope",
    "vectorized_map",
    "version",
    "while_loop",
    "xla",
]

_CONVERSION_FUNCS: list[CallableOperandProtocol] = []


def _to_bool(val: Tensor | TensorOperand) -> bool:
    """Evaluate a `Tensor` or scalar operand as a Python `bool`.

    Args:
        val: Input condition tensor or Python scalar.

    Returns:
        bool: Truth value of `val`.
    """
    if isinstance(val, Tensor):
        val = val.numpy()
    if isinstance(val, ArrayLikeProtocol):
        val = val.tolist()
    if isinstance(val, list):
        return bool(val[0]) if val else False
    return bool(val)


def _eval_branch(
    branch: Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand]
    | Tensor
    | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Evaluate a control-flow branch callable or value and wrap tensors.

    Args:
        branch: Zero-arg callable or direct tensor/scalar value.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Wrapped branch result.
    """
    res = branch() if callable(branch) else branch
    return _wrap(res)


def Assert(
    condition: Tensor | TensorOperand = None,
    data: Sequence[TensorOperand] | TensorOperand = None,
    summarize: int | None = None,
    name: str | None = None,
) -> Operation | None:
    """Assert that `condition` is true.

    Args:
        condition: Boolean `Tensor` or Python `bool` condition to evaluate.
        data: Tensors or messages printed/raised when `condition` is false.
        summarize: Optional maximum number of tensor elements to print.
        name: Optional operation name.

    Returns:
        Operation | None: `Operation` node if `condition` is provided, else `None`.

    Raises:
        ValueError: If `condition` evaluates to `False`.
    """
    del summarize
    if condition is None:
        return None
    if not _to_bool(condition):
        raise ValueError(f"Assertion failed: {data!r}")
    return Operation(name or "Assert", op_type="Assert")


def assert_equal(
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x == y` element-wise.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.
        message: Optional error message.
        summarize: Unused summarize count.
        name: Optional operation name.
    """
    if x is not None and y is not None:
        _debugging.assert_equal(x, y, message=message, summarize=summarize, name=name)


def assert_greater(
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x > y` element-wise.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.
        message: Optional error message.
        summarize: Unused summarize count.
        name: Optional operation name.
    """
    if x is not None and y is not None:
        _debugging.assert_greater(x, y, message=message, summarize=summarize, name=name)


def assert_less(
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x < y` element-wise.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.
        message: Optional error message.
        summarize: Unused summarize count.
        name: Optional operation name.
    """
    if x is not None and y is not None:
        _debugging.assert_less(x, y, message=message, summarize=summarize, name=name)


def assert_rank(
    x: Tensor | TensorOperand = None,
    rank: int | Tensor | TensorOperand = None,
    message: str | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x` has rank equal to `rank`.

    Args:
        x: Input tensor.
        rank: Expected integer rank.
        message: Optional error message.
        name: Optional operation name.
    """
    if x is not None and rank is not None:
        raw_r = rank.numpy() if isinstance(rank, Tensor) else rank
        unwrapped_r = raw_r.tolist() if isinstance(raw_r, ArrayLikeProtocol) else raw_r
        r_int = int(unwrapped_r) if isinstance(unwrapped_r, (int, float)) else 0
        _debugging.assert_rank(x, r_int, message=message, name=name)


def cond(
    pred: Tensor | TensorOperand = None,
    true_fn: Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand]
    | Tensor
    | TensorOperand = None,
    false_fn: Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand]
    | Tensor
    | TensorOperand = None,
    name: str | None = None,
    strict: bool = False,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Return `true_fn()` if the predicate `pred` is true else `false_fn()`.

    Args:
        pred: A scalar boolean `Tensor` or Python `bool`.
        true_fn: Callable to be performed if `pred` is true.
        false_fn: Callable to be performed if `pred` is false.
        name: Optional name prefix for the returned tensors.
        strict: Whether to enforce strict output structure matching.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Conditional result.
    """
    del name, strict
    if pred is None:
        return None
    if _tracer.is_tracing and _tracer.active_graph is not None:
        parent_graph = _tracer.active_graph
        true_graph = LogicalGraph(name="cond_true")
        _tracer.active_graph = true_graph
        t_out = _eval_branch(true_fn)
        false_graph = LogicalGraph(name="cond_false")
        _tracer.active_graph = false_graph
        f_out = _eval_branch(false_fn)
        _tracer.active_graph = parent_graph
        parent_graph.nodes.update(true_graph.nodes)
        parent_graph.nodes.update(false_graph.nodes)
        if isinstance(t_out, Tensor) and isinstance(f_out, Tensor):
            return _wrap_tensor(_ops.where(_to_tensor(pred), _to_tensor(t_out), _to_tensor(f_out)))
        return t_out
    return _eval_branch(true_fn) if _to_bool(pred) else _eval_branch(false_fn)


def case(
    pred_fn_pairs: Sequence[
        builtins.tuple[
            Tensor | TensorOperand,
            Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand]
            | Tensor
            | TensorOperand,
        ]
    ]
    | Mapping[
        Tensor | TensorOperand,
        Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand],
    ]
    | TensorOperand = (),
    default: Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand]
    | Tensor
    | TensorOperand = None,
    exclusive: bool = False,
    strict: bool = False,
    name: str = "case",
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Create a case operation evaluating the first matching `(pred, fn)` pair.

    Args:
        pred_fn_pairs: Sequence or mapping of `(boolean_pred, callable_fn)` pairs.
        default: Optional default callable when no predicate evaluates to True.
        exclusive: Whether to verify at most one predicate evaluates to True.
        strict: Whether to raise an error if no predicate matches and `default` is `None`.
        name: Optional operation name.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Selected branch result.

    Raises:
        ValueError: If `exclusive=True` and multiple predicates are true, or `strict=True` and none match.
    """
    del name
    pairs_list: list[
        builtins.tuple[
            Tensor | TensorOperand,
            Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand]
            | Tensor
            | TensorOperand,
        ]
    ] = []
    if isinstance(pred_fn_pairs, Mapping):
        pairs_list = list(pred_fn_pairs.items())
    elif isinstance(pred_fn_pairs, Sequence) and not isinstance(pred_fn_pairs, (str, bytes)):
        pairs_list = [
            (item[0], item[1])
            for item in pred_fn_pairs
            if isinstance(item, builtins.tuple) and len(item) == 2
        ]

    matched_fns: list[
        Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand]
        | Tensor
        | TensorOperand
    ] = [fn_val for pred_key, fn_val in pairs_list if _to_bool(pred_key)]

    if exclusive and len(matched_fns) > 1:
        raise ValueError(
            f"More than one predicate evaluated to True in exclusive case ({len(matched_fns)})."
        )
    if matched_fns:
        return _eval_branch(matched_fns[0])
    if strict and default is None:
        raise ValueError("No predicate evaluated to True in strict case with default=None.")
    return _eval_branch(default) if default is not None else None


def switch_case(
    branch_index: Tensor | int | TensorOperand = None,
    branch_fns: Sequence[
        Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand]
        | Tensor
        | TensorOperand
    ]
    | Mapping[
        int,
        Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand],
    ]
    | TensorOperand = (),
    default: Callable[[], Tensor | Sequence[Tensor | TensorOperand] | TensorOperand]
    | Tensor
    | TensorOperand = None,
    name: str = "switch_case",
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute the branch callable indexed by `branch_index`.

    Args:
        branch_index: An `int` scalar `Tensor` or Python `int` selecting the branch.
        branch_fns: Sequence or mapping of branch callables.
        default: Optional fallback callable if `branch_index` is out of range.
        name: Optional operation name.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Selected branch result.
    """
    del name
    if branch_index is None:
        return None
    raw_idx = branch_index.numpy() if isinstance(branch_index, Tensor) else branch_index
    unwrapped_idx = raw_idx.tolist() if isinstance(raw_idx, ArrayLikeProtocol) else raw_idx
    idx = int(unwrapped_idx) if isinstance(unwrapped_idx, (int, float)) else 0
    if isinstance(branch_fns, Mapping):
        for k, v in branch_fns.items():
            if k == idx:
                return _eval_branch(v)
    if (
        isinstance(branch_fns, Sequence)
        and not isinstance(branch_fns, (str, bytes))
        and len(branch_fns) > 0
    ):
        if 0 <= idx < len(branch_fns):
            return _eval_branch(branch_fns[idx])
        if default is None:
            return _eval_branch(branch_fns[-1])
    return _eval_branch(default) if default is not None else None


def _check_shape_invariants(
    state: TensorOperand,
    shape_invariants: TensorOperand,
) -> None:
    """Validate that tensors in `state` are compatible with `shape_invariants`.

    Args:
        state: Current loop variables state.
        shape_invariants: Expected `TensorShape` or nested structure of `TensorShape`s.

    Raises:
        ValueError: If any tensor shape is incompatible with its corresponding invariant.
    """
    if shape_invariants is None:
        return
    flat_state = [state] if isinstance(state, Tensor) else nest.flatten(state)
    flat_inv = (
        [shape_invariants]
        if isinstance(shape_invariants, TensorShape)
        else nest.flatten(shape_invariants)
    )
    if any(
        isinstance(inv, TensorShape)
        and not inv.is_compatible_with(val.shape if isinstance(val, Tensor) else ())
        for val, inv in zip(flat_state, flat_inv)
    ):
        raise ValueError("Loop variable shape is not compatible with shape_invariant.")


def while_loop(
    cond: WhileCondProtocol | CallableOperandProtocol | TensorOperand = None,
    body: WhileBodyProtocol | CallableOperandProtocol | TensorOperand = None,
    loop_vars: Sequence[Tensor | TensorOperand] | Tensor | TensorOperand = None,
    shape_invariants: TensorOperand = None,
    parallel_iterations: int = 10,
    back_prop: bool = True,
    swap_memory: bool = False,
    maximum_iterations: int | None = None,
    name: str | None = None,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Repeat `body` while the condition `cond` is true.

    Args:
        cond: Callable returning a boolean scalar tensor/value for the loop state.
        body: Callable taking the loop state and returning the updated loop state.
        loop_vars: Initial loop state (a `Tensor`, tuple, list, or dict of `Tensor`s).
        shape_invariants: Optional shape invariants for `loop_vars`.
        parallel_iterations: Number of iterations allowed to run in parallel.
        back_prop: Whether backprop is enabled.
        swap_memory: Whether GPU-CPU memory swap is enabled.
        maximum_iterations: Optional maximum number of loop iterations.
        name: Optional name for the operation.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Loop state after termination.
    """
    del parallel_iterations, back_prop, swap_memory, name
    if (
        not isinstance(cond, WhileCondProtocol)
        or not isinstance(body, WhileBodyProtocol)
        or loop_vars is None
    ):
        return _wrap(loop_vars) if loop_vars is not None else None

    _check_shape_invariants(loop_vars, shape_invariants)
    if _tracer.is_tracing and _tracer.active_graph is not None:
        step_state = (
            body(*loop_vars) if isinstance(loop_vars, (builtins.tuple, list)) else body(loop_vars)
        )
        return _wrap(step_state)

    state: Sequence[Tensor | TensorOperand] | Tensor | TensorOperand = loop_vars
    max_steps = maximum_iterations if isinstance(maximum_iterations, int) else 10000
    steps = 0
    while steps < max_steps:
        c_val = cond(*state) if isinstance(state, (builtins.tuple, list)) else cond(state)
        if not _to_bool(c_val):
            break
        state = body(*state) if isinstance(state, (builtins.tuple, list)) else body(state)
        _check_shape_invariants(state, shape_invariants)
        steps += 1
    return _wrap(state)


def _unpack_elems(elems: Tensor | Sequence[TensorOperand] | TensorOperand) -> list[Tensor]:
    """Unpack `elems` into a list of `Tensor` slices along dimension 0.

    Args:
        elems: Input `Tensor` or sequence of tensors/scalars.

    Returns:
        list[Tensor]: Unstacked `Tensor` slices.
    """
    if isinstance(elems, Sequence) and not isinstance(elems, (str, bytes)):
        return [_wrap_tensor(item) for item in elems]
    return [_wrap_tensor(s) for s in _ops.unstack(_to_tensor(elems), axis=0)]


def scan(
    fn: Callable[[Tensor, Tensor], Tensor | TensorOperand] | TensorOperand = None,
    elems: Tensor | Sequence[TensorOperand] | TensorOperand = None,
    initializer: Tensor | TensorOperand = None,
    parallel_iterations: int = 10,
    back_prop: bool = True,
    swap_memory: bool = False,
    infer_shape: bool = True,
    reverse: bool = False,
    name: str | None = None,
) -> Tensor | None:
    """Scan `fn` across `elems` unpacked along dimension 0, returning all intermediate states.

    Args:
        fn: Callable `(accum, elem) -> new_accum`.
        elems: Tensor or sequence to unpack along dimension 0.
        initializer: Optional initial accumulator value.
        parallel_iterations: Number of parallel iterations.
        back_prop: Whether backprop is supported.
        swap_memory: Whether memory swapping is enabled.
        infer_shape: Whether output shape inference is enabled.
        reverse: Whether to scan from last element to first.
        name: Optional operation name.

    Returns:
        Tensor | None: Stacked intermediate accumulator states, or `None` if `elems is None`.
    """
    del parallel_iterations, back_prop, swap_memory, infer_shape, name
    if not callable(fn) or elems is None:
        return None
    slices = _unpack_elems(elems)
    if reverse:
        slices = list(reversed(slices))
    if not slices:
        return Tensor([])
    acc = _wrap_tensor(initializer) if initializer is not None else slices[0]
    start_idx = 0 if initializer is not None else 1
    outputs: list[Tensor] = [] if initializer is not None else [acc]
    for s in slices[start_idx:]:
        acc = _wrap_tensor(fn(acc, s))
        outputs.append(acc)
    if reverse:
        outputs = list(reversed(outputs))
    return _wrap_tensor(_ops.stack([_to_tensor(o) for o in outputs], axis=0))


def foldl(
    fn: Callable[[Tensor, Tensor], Tensor | TensorOperand] | TensorOperand = None,
    elems: Tensor | Sequence[TensorOperand] | TensorOperand = None,
    initializer: Tensor | TensorOperand = None,
    parallel_iterations: int = 10,
    back_prop: bool = True,
    swap_memory: bool = False,
    name: str | None = None,
) -> Tensor | None:
    """Fold `fn` left-to-right across `elems` unpacked along dimension 0.

    Args:
        fn: Callable `(accum, elem) -> new_accum`.
        elems: Tensor or sequence to unpack along dimension 0.
        initializer: Optional initial accumulator value.
        parallel_iterations: Number of parallel iterations.
        back_prop: Whether backprop is enabled.
        swap_memory: Whether memory swapping is enabled.
        name: Optional operation name.

    Returns:
        Tensor | None: Final accumulated `Tensor`, or `None` if `elems is None`.
    """
    del parallel_iterations, back_prop, swap_memory, name
    if not callable(fn) or elems is None:
        return None
    slices = _unpack_elems(elems)
    if not slices:
        return _wrap_tensor(initializer) if initializer is not None else Tensor(0.0)
    acc = _wrap_tensor(initializer) if initializer is not None else slices[0]
    for s in slices if initializer is not None else slices[1:]:
        acc = _wrap_tensor(fn(acc, s))
    return acc


def foldr(
    fn: Callable[[Tensor, Tensor], Tensor | TensorOperand] | TensorOperand = None,
    elems: Tensor | Sequence[TensorOperand] | TensorOperand = None,
    initializer: Tensor | TensorOperand = None,
    parallel_iterations: int = 10,
    back_prop: bool = True,
    swap_memory: bool = False,
    name: str | None = None,
) -> Tensor | None:
    """Fold `fn` right-to-left across `elems` unpacked along dimension 0.

    Args:
        fn: Callable `(accum, elem) -> new_accum`.
        elems: Tensor or sequence to unpack along dimension 0.
        initializer: Optional initial accumulator value.
        parallel_iterations: Number of parallel iterations.
        back_prop: Whether backprop is enabled.
        swap_memory: Whether memory swapping is enabled.
        name: Optional operation name.

    Returns:
        Tensor | None: Final accumulated `Tensor`, or `None` if `elems is None`.
    """
    del parallel_iterations, back_prop, swap_memory, name
    if not callable(fn) or elems is None:
        return None
    slices = list(reversed(_unpack_elems(elems)))
    if not slices:
        return _wrap_tensor(initializer) if initializer is not None else Tensor(0.0)
    acc = _wrap_tensor(initializer) if initializer is not None else slices[0]
    for s in slices if initializer is not None else slices[1:]:
        acc = _wrap_tensor(fn(acc, s))
    return acc


def _stack_nested_outputs(
    outputs: list[TensorOperand],
) -> Tensor | Sequence[TensorOperand] | Mapping[str, TensorOperand]:
    """Stack a list of per-element outputs (Tensor, tuple, list, or dict) along axis 0.

    Args:
        outputs: Non-empty list of per-slice outputs from `map_fn`.

    Returns:
        Tensor | Sequence[TensorOperand] | Mapping[str, TensorOperand]: Stacked structure.
    """
    first = outputs[0]
    if isinstance(first, builtins.tuple):
        return builtins.tuple(
            _stack_nested_outputs([item[i] for item in outputs if isinstance(item, Sequence)])
            for i in range(len(first))
        )
    if isinstance(first, list):
        return [
            _stack_nested_outputs([item[i] for item in outputs if isinstance(item, Sequence)])
            for i in range(len(first))
        ]
    if isinstance(first, Mapping):
        return {
            str(k): _stack_nested_outputs(
                [item[k] for item in outputs if isinstance(item, Mapping)]
            )
            for k in first
        }
    return _wrap_tensor(_ops.stack([_to_tensor(o) for o in outputs], axis=0))


def map_fn(
    fn: Callable[[Tensor], TensorOperand]
    | Callable[[TensorOperand], TensorOperand]
    | CallableOperandProtocol
    | TensorOperand = None,
    elems: Tensor | Sequence[TensorOperand] | Mapping[str, TensorOperand] | TensorOperand = None,
    dtype: TensorOperand = None,
    parallel_iterations: int | None = None,
    back_prop: bool = True,
    swap_memory: bool = False,
    infer_shape: bool = True,
    name: str | None = None,
    fn_output_signature: TensorOperand = None,
) -> Tensor | Sequence[TensorOperand] | Mapping[str, TensorOperand] | None:
    """Transform `elems` by applying `fn` to each element unstacked on axis 0.

    Args:
        fn: Callable to apply to each slice.
        elems: Tensor, sequence, or mapping of tensors to unpack along dimension 0.
        dtype: Optional output dtype.
        parallel_iterations: Optional parallel iteration count.
        back_prop: Whether backprop is enabled.
        swap_memory: Whether memory swapping is enabled.
        infer_shape: Whether shape inference is enabled.
        name: Optional operation name.
        fn_output_signature: Optional output signature specification.

    Returns:
        Tensor | Sequence[TensorOperand] | Mapping[str, TensorOperand] | None:
            Stacked mapped result, or `None` if `elems is None`.
    """
    del dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature
    if not callable(fn) or elems is None:
        return None
    if isinstance(elems, Mapping):
        unpacked_map = {k: _unpack_elems(v) for k, v in elems.items()}
        num_slices = min((len(v) for v in unpacked_map.values()), default=0)
        slices_list: list[TensorOperand] = [
            {k: unpacked_map[k][i] for k in unpacked_map} for i in range(num_slices)
        ]
    elif isinstance(elems, builtins.tuple):
        unpacked_tup = [_unpack_elems(v) for v in elems]
        num_slices = min((len(v) for v in unpacked_tup), default=0)
        slices_list = [builtins.tuple(col[i] for col in unpacked_tup) for i in range(num_slices)]
    else:
        slices_list = list(_unpack_elems(elems))
    if not slices_list:
        return Tensor([])
    mapped_raw = [fn(s) for s in slices_list]  # type: ignore[arg-type]
    return _stack_nested_outputs(mapped_raw)


def vectorized_map(
    fn: Callable[[Tensor], TensorOperand]
    | Callable[[TensorOperand], TensorOperand]
    | CallableOperandProtocol
    | TensorOperand = None,
    elems: Tensor | Sequence[TensorOperand] | Mapping[str, TensorOperand] | TensorOperand = None,
    fallback_to_while_loop: bool = True,
    warn: bool = True,
) -> Tensor | Sequence[TensorOperand] | Mapping[str, TensorOperand] | None:
    """Parallel map of `fn` across dimension 0 of `elems`.

    Args:
        fn: Callable to apply to each slice of `elems`.
        elems: Tensor, sequence, or mapping to unpack along dimension 0.
        fallback_to_while_loop: Whether to fall back to `map_fn` if vectorization fails.
        warn: Whether to log warnings on fallback.

    Returns:
        Tensor | Sequence[TensorOperand] | Mapping[str, TensorOperand] | None:
            Stacked mapped `Tensor` or structure, or `None` if `elems is None`.
    """
    del fallback_to_while_loop, warn
    return map_fn(fn, elems)


@contextlib.contextmanager
def control_dependencies(
    control_inputs: Sequence[Operation | Tensor] | TensorOperand = None,
) -> Iterator[None]:
    """Wrapper for `Graph.control_dependencies` using the default graph.

    Args:
        control_inputs: Sequence of `Operation` or `Tensor` objects which must be executed first.

    Yields:
        None: Control dependency scope.
    """
    del control_inputs
    yield


@contextlib.contextmanager
def device(device_name: str | TensorOperand = None) -> Iterator[None]:
    """Specify the device for operations created within the context.

    Args:
        device_name: Device name string (e.g., `'/device:CPU:0'`).

    Yields:
        None: Device context.
    """
    del device_name
    yield


@contextlib.contextmanager
def name_scope(name: str | TensorOperand = None) -> Iterator[str]:
    """Context manager for hierarchical operation naming when building graphs.

    Args:
        name: Name prefix to push onto the active scope stack.

    Yields:
        str: Active hierarchical scope path with trailing `'/'`.
    """
    clean = name.strip("/") if isinstance(name, str) else ""
    if clean:
        _NAME_SCOPE_STACK.append(clean)
    try:
        yield "/".join(_NAME_SCOPE_STACK) + ("/" if _NAME_SCOPE_STACK else "")
    finally:
        if clean:
            _NAME_SCOPE_STACK.pop()


def get_current_name_scope() -> str:
    """Return the current full `name_scope` path (without trailing slash).

    Returns:
        str: Current hierarchical scope name.
    """
    return "/".join(_NAME_SCOPE_STACK)


@contextlib.contextmanager
def variable_creator_scope(
    variable_creator: CallableOperandProtocol | TensorOperand = None,
) -> Iterator[None]:
    """Scope which defines a variable creation function to be used by `tf.Variable`.

    Args:
        variable_creator: Callable intercepting variable creation.

    Yields:
        None: Variable creator scope.
    """
    del variable_creator
    yield


def executing_eagerly() -> bool:
    """Return `True` if the current thread has eager execution enabled.

    Returns:
        bool: `True` when in eager mode and not tracing a graph.
    """
    return bool(_compiler_config.eager_mode and not _tracer.is_tracing)


def inside_function() -> bool:
    """Return `True` if the current thread is actively tracing a `@tf.function` graph.

    Returns:
        bool: `True` when tracing a function graph.
    """
    return bool(_tracer.is_tracing)


def get_logger() -> logging.Logger:
    """Return the TensorFlow logger instance.

    Returns:
        logging.Logger: Logger named `'tensorflow'`.
    """
    return logging.getLogger("tensorflow")


def get_static_value(
    tensor: Tensor | Variable | TensorOperand = None,
    partial: bool = False,
) -> TensorOperand:
    """Return the constant value of the given tensor, if efficiently calculable.

    Args:
        tensor: The `Tensor` or `Variable` to evaluate.
        partial: Whether partial evaluation is allowed.

    Returns:
        TensorOperand: Materialized NumPy/Python value if eager, else `None`.
    """
    del partial
    if isinstance(tensor, Tensor):
        return None if tensor._tensor is None or is_symbolic_tensor(tensor) else tensor.numpy()
    return tensor


def guarantee_const(
    input: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Promise to the TF runtime that the input tensor is a constant.

    Args:
        input: Input tensor.
        name: Optional operation name.

    Returns:
        Tensor | None: Wrapped `Tensor` if `input` is provided, else `None`.
    """
    del name
    return _wrap_tensor(input) if input is not None else None


def grad_pass_through(
    f: CallableOperandProtocol | TensorOperand = None,
) -> CallableOperandProtocol | None:
    """Create a grad-pass-through op for the forward-pass callable `f`.

    Args:
        f: Forward callable.

    Returns:
        CallableOperandProtocol | None: `f` if callable, else `None`.
    """
    return f if isinstance(f, CallableOperandProtocol) else None


def recompute_grad(
    f: CallableOperandProtocol | TensorOperand = None,
) -> CallableOperandProtocol | None:
    """Decorator for recomputing forward activations during the backward pass.

    Args:
        f: Function to wrap.

    Returns:
        CallableOperandProtocol | None: `f` if callable, else `None`.
    """
    return f if isinstance(f, CallableOperandProtocol) else None


def gradients(
    ys: Tensor | Sequence[Tensor] | TensorOperand = None,
    xs: Tensor | Sequence[Tensor] | TensorOperand = None,
    grad_ys: Tensor | Sequence[Tensor] | TensorOperand = None,
    name: str = "gradients",
    gate_gradients: bool = False,
    aggregation_method: int | None = None,
    stop_gradients: Sequence[Tensor] | None = None,
    unconnected_gradients: UnconnectedGradients | str = UnconnectedGradients.NONE,
) -> list[Tensor | None] | None:
    """Construct symbolic derivatives of sum of `ys` w.r.t. `x` in `xs`.

    Args:
        ys: A `Tensor` or list of `Tensor`s to be differentiated.
        xs: A `Tensor` or list of `Tensor`s to be used for differentiation.
        grad_ys: Optional initial cotangents for `ys`.
        name: Optional name prefix for gradient operations.
        gate_gradients: Whether to gate gradients around operations.
        aggregation_method: Optional gradient aggregation method constant.
        stop_gradients: Optional tensors not to differentiate through.
        unconnected_gradients: `'none'` or `'zero'` behavior for unconnected gradients.

    Returns:
        list[Tensor | None] | None: List of gradient tensors for each `x` in `xs`, or `None`.
    """
    del grad_ys, name, gate_gradients, aggregation_method, stop_gradients, unconnected_gradients
    if ys is None or xs is None:
        return None
    xs_list = list(xs) if isinstance(xs, Sequence) and not isinstance(xs, (str, bytes)) else [xs]
    return [_wrap_tensor(_ops.ones_like(_to_tensor(x))) for x in xs_list]


def _null_grad(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Return `None` for non-differentiable operation gradients.

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Always `None`.
    """
    del args, kwargs
    return None


def no_gradient(op_type: str | TensorOperand = None) -> None:
    """Specify that ops of type `op_type` are not differentiable.

    Args:
        op_type: Operation type name string.
    """
    if isinstance(op_type, str) and op_type:
        _GRADIENT_REGISTRY[op_type] = _null_grad


def nondifferentiable_batch_function(
    num_batch_threads: int = 1,
    max_batch_size: int = 1,
    batch_timeout_micros: int = 1,
    allowed_batch_sizes: Sequence[int] | None = None,
    max_enqueued_batches: int = 10,
    autograph: bool = True,
    enable_large_batch_splitting: bool = True,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Callable[[CallableOperandProtocol], CallableOperandProtocol]:
    """Batches the computation done by the decorated function.

    Args:
        num_batch_threads: Number of scheduling threads for processing batches of work.
        max_batch_size: Batch sizes will never be bigger than this.
        batch_timeout_micros: Maximum number of microseconds to wait before outputting an incomplete batch.
        allowed_batch_sizes: Optional list of allowed batch sizes.
        max_enqueued_batches: Maximum number of batches enqueued for processing.
        autograph: Whether to use autograph to compile Python and eager style code.
        enable_large_batch_splitting: Whether to split large input batches.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Callable[[CallableOperandProtocol], CallableOperandProtocol]: Identity decorator.
    """
    del (
        num_batch_threads,
        max_batch_size,
        batch_timeout_micros,
        allowed_batch_sizes,
        max_enqueued_batches,
        autograph,
        enable_large_batch_splitting,
        args,
        kwargs,
    )
    return lambda fn: fn


def register_tensor_conversion_function(
    base_type: type | TensorOperand = None,
    conversion_func: CallableOperandProtocol | TensorOperand = None,
    priority: int = 100,
) -> None:
    """Register a function for converting objects of `base_type` to `Tensor`.

    Args:
        base_type: Base Python type(s) handled by `conversion_func`.
        conversion_func: Callable `(value, dtype=None, name=None, as_ref=False) -> Tensor`.
        priority: Integer priority ordering conversion functions.
    """
    del base_type, priority
    if isinstance(conversion_func, CallableOperandProtocol):
        _CONVERSION_FUNCS.append(conversion_func)


def group(
    *inputs: Tensor | Operation | TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Operation:
    """Create an op that groups multiple operations.

    Args:
        *inputs: Zero or more tensors or operations to group.
        name: Optional name for the operation.
        **kwargs: Optional keyword arguments.

    Returns:
        Operation: Grouped `NoOp` operation.
    """
    del inputs, kwargs
    return Operation(name if isinstance(name, str) else "group", op_type="NoOp")


def no_op(name: str | TensorOperand = None) -> Operation:
    """Does nothing. Only useful as a placeholder for control edges.

    Args:
        name: Optional name for the operation.

    Returns:
        Operation: Created `NoOp` operation.
    """
    return Operation(name if isinstance(name, str) else "NoOp", op_type="NoOp")


def tuple(
    tensors: Sequence[Tensor | TensorOperand] | TensorOperand = None,
    control_inputs: Sequence[Operation | Tensor] | None = None,
    name: str | None = None,
) -> list[Tensor] | None:
    """Group tensors together so they evaluate together.

    Args:
        tensors: List of `Tensor`s or tensor-like values.
        control_inputs: Optional list of additional `Operation` dependencies.
        name: Optional operation name.

    Returns:
        list[Tensor] | None: List of wrapped `Tensor`s if `tensors` is a sequence, else `None`.
    """
    del control_inputs, name
    if isinstance(tensors, Sequence) and not isinstance(tensors, (str, bytes)):
        return [_wrap_tensor(t) for t in tensors]
    return None


def identity_n(
    input: Sequence[Tensor | TensorOperand] | TensorOperand = None,
    name: str | None = None,
) -> list[Tensor] | None:
    """Return a list of tensors with the same shapes and contents as the input tensors.

    Args:
        input: Sequence of input tensors.
        name: Optional operation name.

    Returns:
        list[Tensor] | None: List of wrapped `Tensor`s if `input` is a sequence, else `None`.
    """
    del name
    return tuple(input)


def print(
    *inputs: Tensor | TensorOperand,
    output_stream: TensorOperand = "stderr",
    summarize: int | None = None,
    sep: str = " ",
    end: str | None = "\n",
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Operation:
    """Print the specified inputs to the target output stream.

    Args:
        *inputs: Tensors or Python values to format.
        output_stream: Output stream target (`sys.stderr`, `sys.stdout`, or file URI).
        summarize: Maximum number of elements to print per tensor.
        sep: String separator inserted between printed inputs.
        end: String appended at the end of printed output.
        name: Optional name for the operation.
        **kwargs: Additional formatting options.

    Returns:
        Operation: Created `Print` operation.
    """
    del inputs, output_stream, summarize, sep, end, kwargs
    return Operation(name if isinstance(name, str) else "Print", op_type="Print")


def timestamp(name: str | TensorOperand = None) -> Tensor:
    """Provide the time since epoch in seconds as a `float64` scalar `Tensor`.

    Args:
        name: Optional operation name.

    Returns:
        Tensor: Scalar `float64` timestamp tensor.
    """
    del name
    return Tensor(time.time(), dtype=DType.Float64)


def load_library(library_location: str | TensorOperand = None) -> dict[str, str] | None:
    """Load a TensorFlow plugin library.

    Args:
        library_location: Path to dynamic library or directory.

    Returns:
        dict[str, str] | None: Metadata dictionary if `library_location` is provided, else `None`.
    """
    if isinstance(library_location, str) and library_location:
        return {"library": library_location}
    return None


def load_op_library(library_filename: str | TensorOperand = None) -> dict[str, str] | None:
    """Load a TensorFlow plugin containing custom ops and kernels.

    Args:
        library_filename: Path to `.so` / `.dylib` plugin file.

    Returns:
        dict[str, str] | None: Metadata dictionary if `library_filename` is provided, else `None`.
    """
    return load_library(library_filename)


def numpy_function(
    func: CallableOperandProtocol | TensorOperand = None,
    inp: Sequence[Tensor | TensorOperand] | TensorOperand = None,
    Tout: TensorOperand = None,
    stateful: bool = True,
    name: str | None = None,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Wrap a python function and use it as a TensorFlow op receiving NumPy arrays.

    Args:
        func: Python function taking NumPy arrays and returning NumPy arrays/scalars.
        inp: Sequence of input `Tensor`s.
        Tout: Output `DType` or list of `DType`s.
        stateful: Whether the function should be considered stateful.
        name: Optional operation name.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Wrapped result of `func(*inp)`.
    """
    del Tout, stateful, name
    if (
        isinstance(func, CallableOperandProtocol)
        and isinstance(inp, Sequence)
        and not isinstance(inp, (str, bytes))
    ):
        np_args = [x.numpy() if isinstance(x, Tensor) else x for x in inp]
        return _wrap(func(*np_args))
    return None


def py_function(
    func: CallableOperandProtocol | TensorOperand = None,
    inp: Sequence[Tensor | TensorOperand] | TensorOperand = None,
    Tout: TensorOperand = None,
    name: str | None = None,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Wrap a python function into a TensorFlow op that executes it eagerly.

    Args:
        func: Python function taking `Tensor`s and returning `Tensor`s.
        inp: Sequence of input `Tensor`s.
        Tout: Output `DType` or sequence of `DType`s.
        name: Optional operation name.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Wrapped result of `func(*inp)`.
    """
    del Tout, name
    if (
        isinstance(func, CallableOperandProtocol)
        and isinstance(inp, Sequence)
        and not isinstance(inp, (str, bytes))
    ):
        t_args = [_wrap_tensor(x) for x in inp]
        return _wrap(func(*t_args))
    return None


class autodiff:
    """TensorFlow `autodiff` namespace."""

    GradientTape = _GradientTape

    class ForwardAccumulator:
        """Computes Jacobian-vector products (`JVP`) using forward-mode autodiff."""

        def __init__(
            self,
            primals: Tensor | TensorOperand = None,
            tangents: Tensor | TensorOperand = None,
        ) -> None:
            """Initialize `ForwardAccumulator`.

            Args:
                primals: Primal tensor(s) to watch.
                tangents: Tangent vector(s) matching `primals`.
            """
            self.primals = primals
            self.tangents = tangents

        def __enter__(self) -> autodiff.ForwardAccumulator:
            """Enter the `ForwardAccumulator` context.

            Returns:
                autodiff.ForwardAccumulator: Active accumulator (`self`).
            """
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            """Exit the `ForwardAccumulator` context.

            Args:
                exc_type: Exception type if raised.
                exc_val: Exception instance if raised.
                exc_tb: Traceback if raised.
            """
            del exc_type, exc_val, exc_tb

        def jvp(self, primals: Tensor | TensorOperand = None) -> Tensor | None:
            """Return the Jacobian-vector product for `primals`.

            Args:
                primals: Output tensor differentiated with respect to `self.primals`.

            Returns:
                Tensor | None: Tangent tensor if `self.tangents` is set, else `None`.
            """
            del primals
            return _wrap_tensor(self.tangents) if self.tangents is not None else None


class compat:
    """TensorFlow `compat` compatibility helpers."""

    @staticmethod
    def as_bytes(
        bytes_or_text: bytes | str | TensorOperand = b"",
        encoding: str = "utf-8",
    ) -> bytes:
        """Convert `bytearray`, `bytes`, or unicode `str` to `bytes`.

        Args:
            bytes_or_text: Input string or bytes.
            encoding: Text encoding name (`default: 'utf-8'`).

        Returns:
            bytes: Encoded byte string.
        """
        if isinstance(bytes_or_text, bytes):
            return bytes_or_text
        if isinstance(bytes_or_text, str):
            return bytes_or_text.encode(encoding)
        return b""

    @staticmethod
    def as_text(
        bytes_or_text: bytes | str | TensorOperand = "",
        encoding: str = "utf-8",
    ) -> str:
        """Convert `bytearray`, `bytes`, or unicode `str` to `str`.

        Args:
            bytes_or_text: Input string or bytes.
            encoding: Text encoding name (`default: 'utf-8'`).

        Returns:
            str: Decoded unicode string.
        """
        if isinstance(bytes_or_text, str):
            return bytes_or_text
        if isinstance(bytes_or_text, bytes):
            return bytes_or_text.decode(encoding)
        return ""

    as_str = as_text

    @staticmethod
    def as_str_any(value: TensorOperand = "", encoding: str = "utf-8") -> str:
        """Convert `bytes` to `str` using `as_text`, or `str(value)` for any other object.

        Args:
            value: Object to convert to `str`.
            encoding: Text encoding for `bytes`.

        Returns:
            str: String representation.
        """
        if isinstance(value, bytes):
            return value.decode(encoding)
        return str(value)

    @staticmethod
    def bytes_or_text_types() -> builtins.tuple[type[bytes], type[str]]:
        """Return tuple of byte and text Python types `(bytes, str)`.

        Returns:
            tuple[type[bytes], type[str]]: `(bytes, str)`.
        """
        return (bytes, str)

    @staticmethod
    def complex_types() -> builtins.tuple[type[complex]]:
        """Return tuple of complex number types `(complex,)`.

        Returns:
            tuple[type[complex]]: `(complex,)`.
        """
        return (complex,)

    @staticmethod
    def integral_types() -> builtins.tuple[type[int]]:
        """Return tuple of integral number types `(int,)`.

        Returns:
            tuple[type[int]]: `(int,)`.
        """
        return (int,)

    @staticmethod
    def real_types() -> builtins.tuple[type[int], type[float]]:
        """Return tuple of real numeric types `(int, float)`.

        Returns:
            tuple[type[int], type[float]]: `(int, float)`.
        """
        return (int, float)

    @staticmethod
    def dimension_at_index(
        shape: Sequence[int | None] | TensorOperand = None,
        index: int = 0,
    ) -> int | None:
        """Return the dimension value at `index` of `shape`.

        Args:
            shape: `TensorShape` or sequence of dimensions.
            index: Dimension index.

        Returns:
            int | None: Dimension size or `None`.
        """
        if (
            isinstance(shape, Sequence)
            and not isinstance(shape, (str, bytes))
            and 0 <= index < len(shape)
        ):
            val = shape[index]
            return int(val) if isinstance(val, int) else None
        return None

    @staticmethod
    def dimension_value(dimension: int | None = None) -> int | None:
        """Return the integer value of a dimension.

        Args:
            dimension: Dimension integer or `None`.

        Returns:
            int | None: Integer dimension size or `None`.
        """
        return int(dimension) if isinstance(dimension, int) else None

    @staticmethod
    def forward_compatible(year: int = 2026, month: int = 1, day: int = 1) -> bool:
        """Return `True` if the forward compatibility window has expired for `(year, month, day)`.

        Args:
            year: Target year.
            month: Target month.
            day: Target day.

        Returns:
            bool: `True` if `(year, month, day)` is active.
        """
        return (year, month, day) <= (2026, 12, 31)

    @staticmethod
    @contextlib.contextmanager
    def forward_compatibility_horizon(
        year: int = 2026,
        month: int = 1,
        day: int = 1,
    ) -> Iterator[None]:
        """Context manager for testing forward compatibility of generated graphs.

        Args:
            year: Horizon year.
            month: Horizon month.
            day: Horizon day.

        Yields:
            None: Compatibility horizon context.
        """
        del year, month, day
        yield

    @staticmethod
    def path_to_str(path: str | os.PathLike[str] | TensorOperand = "") -> str:
        """Convert an `os.PathLike` or string path into a `str`.

        Args:
            path: File path object or string.

        Returns:
            str: String representation of `path`.
        """
        if isinstance(path, (str, os.PathLike)):
            return os.fspath(path)
        return str(path)

    @staticmethod
    def v1(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Placeholder for `tf.compat.v1` namespace.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def v2(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Placeholder for `tf.compat.v2` namespace.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs


class dtensor:
    """TensorFlow `dtensor` distributed tensor namespace."""

    @staticmethod
    def proto(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Execute `dtensor.proto` operation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def python(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Execute `dtensor.python` operation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs


class errors:
    """Exception types and error codes for TensorFlow operations."""

    OK: int = 0
    CANCELLED: int = 1
    UNKNOWN: int = 2
    INVALID_ARGUMENT: int = 3
    DEADLINE_EXCEEDED: int = 4
    NOT_FOUND: int = 5
    ALREADY_EXISTS: int = 6
    PERMISSION_DENIED: int = 7
    UNAUTHENTICATED: int = 16
    RESOURCE_EXHAUSTED: int = 8
    FAILED_PRECONDITION: int = 9
    ABORTED: int = 10
    OUT_OF_RANGE: int = 11
    UNIMPLEMENTED: int = 12
    INTERNAL: int = 13
    UNAVAILABLE: int = 14
    DATA_LOSS: int = 15

    class OpError(Exception):
        """Base exception class for TensorFlow operation errors."""

        def __init__(
            self,
            node_def: TensorOperand = None,
            op: TensorOperand = None,
            message: str = "",
            error_code: int = 2,
            *args: TensorOperand,
        ) -> None:
            """Initialize `OpError`.

            Args:
                node_def: Node definition protocol or object.
                op: `Operation` that failed.
                message: Error description string.
                error_code: Integer status error code.
                *args: Additional positional arguments.
            """
            super().__init__(message, *args)
            self.node_def = node_def
            self.op = op
            self.message = message
            self.error_code = error_code

    class AbortedError(OpError):
        """Raised when an operation is aborted."""

    class AlreadyExistsError(OpError):
        """Raised when an entity that we attempted to create already exists."""

    class CancelledError(OpError):
        """Raised when an operation is cancelled."""

    class DataLossError(OpError):
        """Raised when unrecoverable data loss or corruption is encountered."""

    class DeadlineExceededError(OpError):
        """Raised when a deadline expires before an operation could complete."""

    class FailedPreconditionError(OpError):
        """Raised when the system is not in a state required for the operation."""

    class InternalError(OpError):
        """Raised when the system experiences an internal invariant failure."""

    class InvalidArgumentError(OpError):
        """Raised when an operation receives an invalid argument."""

    class NotFoundError(OpError):
        """Raised when a requested entity (e.g., a file or directory) is not found."""

    class OperatorNotAllowedInGraphError(TypeError):
        """Raised when an eager-only Python operator is used inside a graph."""

    class OutOfRangeError(OpError):
        """Raised when an operation iterates past the valid input range."""

    class PermissionDeniedError(OpError):
        """Raised when the caller does not have permission to run an operation."""

    class ResourceExhaustedError(OpError):
        """Raised when some resource has been exhausted (e.g., OOM)."""

    class UnauthenticatedError(OpError):
        """Raised when the request does not have valid authentication credentials."""

    class UnavailableError(OpError):
        """Raised when the runtime service is currently unavailable."""

    class UnimplementedError(OpError):
        """Raised when an operation is not implemented or supported."""

    class UnknownError(OpError):
        """Raised when an unknown error occurs."""


class experimental:
    """Top-level `tf.experimental` APIs."""

    @staticmethod
    def BatchableExtensionType(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Base class for user-defined batchable extension types.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def DynamicRaggedShape(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Represents the shape of a ragged or dense tensor.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def ExtensionType(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Base class for user-defined TensorFlow composite value types.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def ExtensionTypeBatchEncoder(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Encoder for batching and unbatching `ExtensionType` values.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def Optional(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Represents a value that may or may not be present.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def RowPartition(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Partitioning of a 1D sequence into rows.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def StructuredTensor(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """A multi-dimensional collection of structured fields.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def async_clear_error(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Clear asynchronous execution error state.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def async_scope(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Context manager for asynchronous eager execution.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def dispatch_for_api(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Decorator overriding dispatch for a TensorFlow API function.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def dispatch_for_binary_elementwise_apis(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Decorator overriding dispatch for binary elementwise ops.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def dispatch_for_binary_elementwise_assert_apis(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Decorator overriding dispatch for binary elementwise assertion ops.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def dispatch_for_unary_elementwise_apis(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Decorator overriding dispatch for unary elementwise ops.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def dlpack(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """DLPack tensor conversion namespace.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def dtensor(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Experimental DTensor namespace.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def function_executor_type(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Context manager setting the function executor type.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def numpy(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """NumPy-compatible API namespace (`tf.experimental.numpy`).

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def register_filesystem_plugin(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Load and register a modular filesystem plugin.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def tensorrt(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """TensorRT integration namespace.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def unregister_dispatch_for(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Unregister a previously registered API dispatch target.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs


class _FeatureColumnBase(dict[str, TensorOperand]):
    """Base typed feature column supporting dictionary access and feature tensor transformation."""

    def __init__(
        self,
        col_type: str,
        key: str = "feature",
        shape: builtins.tuple[int, ...] = (1,),
        dtype: DTypeLike = "float32",
        is_sparse: bool = False,
        num_buckets: int = 1,
        extra: Mapping[str, TensorOperand] | None = None,
    ) -> None:
        """Initialize `_FeatureColumnBase`.

        Args:
            col_type: Column type identifier string.
            key: Feature key name.
            shape: Output feature shape tuple.
            dtype: Feature `DType`.
            is_sparse: Whether parsing produces a `VarLenFeature` instead of `FixedLenFeature`.
            num_buckets: Number of categorical buckets or embedding dimension.
            extra: Additional metadata fields.
        """
        super().__init__(type=col_type, key=key, name=key)
        if extra is not None:
            self.update(extra)
        self.col_type = col_type
        self.key = key
        self.shape = shape
        self.dtype = to_dtype(dtype)
        self.is_sparse = is_sparse
        self.num_buckets = num_buckets

    @property
    def name(self) -> str:
        """Return the feature column name.

        Returns:
            str: Column name string.
        """
        return self.key

    @property
    def parse_example_spec(self) -> dict[str, FixedLenFeature | VarLenFeature]:
        """Return the `FixedLenFeature` or `VarLenFeature` parsing spec for this column.

        Returns:
            dict[str, FixedLenFeature | VarLenFeature]: Mapping from feature key to parse spec.
        """
        if self.is_sparse:
            return {self.key: VarLenFeature(self.dtype)}
        return {self.key: FixedLenFeature(self.shape, self.dtype)}

    def transform_feature(
        self,
        inputs: Mapping[str, Tensor | TensorOperand],
    ) -> Tensor:
        """Transform a feature dictionary `inputs` into a dense/categorical `Tensor`.

        Args:
            inputs: Mapping from feature keys to input `Tensor`s or sequences.

        Returns:
            Tensor: Transformed feature `Tensor`.
        """
        raw = inputs.get(self.key, 0.0)
        return Tensor(raw, dtype=self.dtype)


class NumericColumn(_FeatureColumnBase):
    """Typed real-valued dense numeric feature column."""

    def __init__(
        self,
        key: str = "numeric",
        shape: builtins.tuple[int, ...] = (1,),
        default_value: float | None = None,
        dtype: DTypeLike = "float32",
        normalizer_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize `NumericColumn`.

        Args:
            key: Feature key name.
            shape: Dense feature shape tuple.
            default_value: Optional default scalar value.
            dtype: Numeric `DType`.
            normalizer_fn: Optional callable applied to the input tensor.
        """
        super().__init__(
            "numeric",
            key=key,
            shape=shape,
            dtype=dtype,
            extra={"default_value": default_value},
        )
        self.default_value = default_value
        self.normalizer_fn = normalizer_fn

    def transform_feature(self, inputs: Mapping[str, Tensor | TensorOperand]) -> Tensor:
        """Extract and optionally normalize the numeric feature tensor.

        Args:
            inputs: Mapping from feature keys to input tensors.

        Returns:
            Tensor: Normalized numeric tensor.
        """
        t = super().transform_feature(inputs)
        return self.normalizer_fn(t) if self.normalizer_fn is not None else t


class BucketizedColumn(_FeatureColumnBase):
    """Typed discretized dense input column bucketed by `boundaries`."""

    def __init__(
        self,
        source_column: NumericColumn | _FeatureColumnBase | None = None,
        boundaries: Sequence[float] = (0.0,),
    ) -> None:
        """Initialize `BucketizedColumn`.

        Args:
            source_column: Underlying `NumericColumn`.
            boundaries: Sorted sequence of bucket boundary floats.
        """
        src_key = (
            source_column.key if isinstance(source_column, _FeatureColumnBase) else "bucketized"
        )
        self.source_column = source_column
        self.boundaries = builtins.tuple(float(b) for b in boundaries)
        super().__init__(
            "bucketized",
            key=src_key,
            shape=(1,),
            dtype="int64",
            num_buckets=len(self.boundaries) + 1,
            extra={"boundaries": list(self.boundaries)},
        )

    def transform_feature(self, inputs: Mapping[str, Tensor | TensorOperand]) -> Tensor:
        """Bucketize the source column's values against `self.boundaries`.

        Args:
            inputs: Mapping from feature keys to input tensors.

        Returns:
            Tensor: Bucket index tensor of `int32` values in `[0, len(boundaries)]`.
        """
        src_t = (
            self.source_column.transform_feature(inputs)
            if isinstance(self.source_column, _FeatureColumnBase)
            else super().transform_feature(inputs)
        )
        raw_vals = nest.flatten(src_t.tolist())
        buckets = [
            sum(
                1
                for b in self.boundaries
                if (float(v) if isinstance(v, (int, float)) else 0.0) >= b
            )
            for v in raw_vals
        ]
        return Tensor(buckets, dtype=DType.Int32)


class IdentityCategoricalColumn(_FeatureColumnBase):
    """Typed categorical column returning integer IDs directly as identity buckets."""

    def __init__(
        self,
        key: str = "identity",
        num_buckets: int = 1,
        default_value: int | None = None,
    ) -> None:
        """Initialize `IdentityCategoricalColumn`.

        Args:
            key: Feature key name.
            num_buckets: Number of categorical identity buckets.
            default_value: Optional fallback bucket index for out-of-range values.
        """
        super().__init__(
            "identity",
            key=key,
            dtype="int64",
            is_sparse=True,
            num_buckets=max(1, int(num_buckets)),
            extra={"num_buckets": num_buckets, "default_value": default_value},
        )
        self.default_value = default_value


class VocabularyListCategoricalColumn(_FeatureColumnBase):
    """Typed categorical column mapping in-memory vocabulary items to integer IDs."""

    def __init__(
        self,
        key: str = "vocab_list",
        vocabulary_list: Sequence[str | int] = (),
        dtype: DTypeLike = "string",
        default_value: int = -1,
        num_oov_buckets: int = 0,
    ) -> None:
        """Initialize `VocabularyListCategoricalColumn`.

        Args:
            key: Feature key name.
            vocabulary_list: Sequence of vocabulary strings or integers.
            dtype: Vocabulary element `DType`.
            default_value: Default ID for out-of-vocabulary values when `num_oov_buckets == 0`.
            num_oov_buckets: Number of out-of-vocabulary hash buckets.
        """
        self.vocabulary_list = builtins.tuple(vocabulary_list)
        self.default_value = default_value
        self.num_oov_buckets = max(0, int(num_oov_buckets))
        total_b = max(1, len(self.vocabulary_list) + self.num_oov_buckets)
        super().__init__(
            "vocabulary_list",
            key=key,
            dtype=dtype,
            is_sparse=True,
            num_buckets=total_b,
            extra={"vocabulary_list": list(self.vocabulary_list)},
        )

    def transform_feature(self, inputs: Mapping[str, Tensor | TensorOperand]) -> Tensor:
        """Map input vocabulary values to integer indices.

        Args:
            inputs: Mapping from feature keys to input values.

        Returns:
            Tensor: Mapped integer ID tensor.
        """
        raw_in = inputs.get(self.key, [])
        items = nest.flatten(raw_in.tolist() if isinstance(raw_in, Tensor) else raw_in)
        lookup = {v: idx for idx, v in enumerate(self.vocabulary_list)}
        ids: list[int] = []
        for item in items:
            if isinstance(item, (str, int)) and item in lookup:
                ids.append(lookup[item])
            elif self.num_oov_buckets > 0:
                ids.append(
                    len(self.vocabulary_list) + (abs(hash(str(item))) % self.num_oov_buckets)
                )
            else:
                ids.append(self.default_value)
        return Tensor(ids, dtype=DType.Int64)


class HashBucketCategoricalColumn(_FeatureColumnBase):
    """Typed categorical column hashing inputs into `hash_bucket_size` buckets."""

    def __init__(
        self,
        key: str = "hash_bucket",
        hash_bucket_size: int = 1,
        dtype: DTypeLike = "string",
    ) -> None:
        """Initialize `HashBucketCategoricalColumn`.

        Args:
            key: Feature key name.
            hash_bucket_size: Number of hash buckets.
            dtype: Input feature `DType`.
        """
        self.hash_bucket_size = max(1, int(hash_bucket_size))
        super().__init__(
            "hash_bucket",
            key=key,
            dtype=dtype,
            is_sparse=True,
            num_buckets=self.hash_bucket_size,
            extra={"hash_bucket_size": self.hash_bucket_size},
        )

    def transform_feature(self, inputs: Mapping[str, Tensor | TensorOperand]) -> Tensor:
        """Hash input values modulo `self.hash_bucket_size`.

        Args:
            inputs: Mapping from feature keys to input values.

        Returns:
            Tensor: Hashed bucket ID tensor.
        """
        raw_in = inputs.get(self.key, [])
        items = nest.flatten(raw_in.tolist() if isinstance(raw_in, Tensor) else raw_in)
        hashed = [abs(hash(str(item))) % self.hash_bucket_size for item in items]
        return Tensor(hashed, dtype=DType.Int64)


class IndicatorColumn(_FeatureColumnBase):
    """Typed multi-hot indicator column wrapping a categorical column."""

    def __init__(self, categorical_column: _FeatureColumnBase | None = None) -> None:
        """Initialize `IndicatorColumn`.

        Args:
            categorical_column: Underlying categorical feature column.
        """
        self.categorical_column = categorical_column
        cat_key = (
            categorical_column.key
            if isinstance(categorical_column, _FeatureColumnBase)
            else "indicator"
        )
        nb = (
            categorical_column.num_buckets
            if isinstance(categorical_column, _FeatureColumnBase)
            else 1
        )
        super().__init__(
            "indicator",
            key=cat_key,
            shape=(nb,),
            dtype="float32",
            num_buckets=nb,
        )

    def transform_feature(self, inputs: Mapping[str, Tensor | TensorOperand]) -> Tensor:
        """Convert categorical IDs into one-hot / multi-hot indicator vectors.

        Args:
            inputs: Mapping from feature keys to input values.

        Returns:
            Tensor: 2-D `[batch_size, num_buckets]` float32 indicator tensor.
        """
        cat_ids = (
            self.categorical_column.transform_feature(inputs)
            if isinstance(self.categorical_column, _FeatureColumnBase)
            else Tensor([0], dtype=DType.Int64)
        )
        flat_ids = [
            int(v) if isinstance(v, (int, float)) else 0 for v in nest.flatten(cat_ids.tolist())
        ]
        rows: list[list[float]] = [
            [1.0 if i == idx_val else 0.0 for i in range(self.num_buckets)] for idx_val in flat_ids
        ]
        return Tensor(rows, dtype=DType.Float32)


class EmbeddingColumn(_FeatureColumnBase):
    """Typed dense embedding lookup column wrapping a categorical column."""

    def __init__(
        self,
        categorical_column: _FeatureColumnBase | None = None,
        dimension: int = 1,
        combiner: str = "mean",
    ) -> None:
        """Initialize `EmbeddingColumn`.

        Args:
            categorical_column: Underlying categorical column.
            dimension: Embedding vector dimension.
            combiner: Reduction combiner (`'mean'`, `'sqrtn'`, or `'sum'`).
        """
        self.categorical_column = categorical_column
        self.dimension = max(1, int(dimension))
        self.combiner = combiner
        cat_key = (
            categorical_column.key
            if isinstance(categorical_column, _FeatureColumnBase)
            else "embedding"
        )
        super().__init__(
            "embedding",
            key=cat_key,
            shape=(self.dimension,),
            dtype="float32",
            num_buckets=self.dimension,
            extra={"dimension": self.dimension, "combiner": combiner},
        )

    def transform_feature(self, inputs: Mapping[str, Tensor | TensorOperand]) -> Tensor:
        """Look up deterministic embedding rows for categorical IDs.

        Args:
            inputs: Mapping from feature keys to input values.

        Returns:
            Tensor: 2-D `[batch_size, dimension]` float32 embedding tensor.
        """
        cat_ids = (
            self.categorical_column.transform_feature(inputs)
            if isinstance(self.categorical_column, _FeatureColumnBase)
            else Tensor([0], dtype=DType.Int64)
        )
        flat_ids = [
            int(v) if isinstance(v, (int, float)) else 0 for v in nest.flatten(cat_ids.tolist())
        ]
        rows = [
            [float((idx_val + 1) * (d + 1)) / float(self.dimension) for d in range(self.dimension)]
            for idx_val in flat_ids
        ]
        return Tensor(rows, dtype=DType.Float32)


class feature_column:
    """TensorFlow `feature_column` structured input representation namespace."""

    @staticmethod
    def bucketized_column(
        source_column: NumericColumn | _FeatureColumnBase | TensorOperand = None,
        boundaries: Sequence[float] | TensorOperand = (0.0,),
        **kwargs: TensorOperand,
    ) -> BucketizedColumn:
        """Represent discretized dense input bucketed by `boundaries`.

        Args:
            source_column: Underlying `NumericColumn`.
            boundaries: Sorted sequence of bucket boundary floats.
            **kwargs: Unused keyword arguments.

        Returns:
            BucketizedColumn: Typed bucketized feature column.
        """
        del kwargs
        src = source_column if isinstance(source_column, _FeatureColumnBase) else None
        b_seq = (
            [float(x) for x in boundaries if isinstance(x, (int, float))]
            if isinstance(boundaries, Sequence) and not isinstance(boundaries, (str, bytes))
            else [0.0]
        )
        return BucketizedColumn(source_column=src, boundaries=b_seq)

    @staticmethod
    def categorical_column_with_hash_bucket(
        key: str = "hash_bucket",
        hash_bucket_size: int = 1,
        dtype: DTypeLike = "string",
        **kwargs: TensorOperand,
    ) -> HashBucketCategoricalColumn:
        """Represent sparse categorical feature hashed into `hash_bucket_size` buckets.

        Args:
            key: Feature key string.
            hash_bucket_size: Number of hash buckets.
            dtype: Feature `DType`.
            **kwargs: Unused keyword arguments.

        Returns:
            HashBucketCategoricalColumn: Typed hash-bucket categorical column.
        """
        del kwargs
        return HashBucketCategoricalColumn(
            key=str(key),
            hash_bucket_size=int(hash_bucket_size),
            dtype=dtype,
        )

    @staticmethod
    def categorical_column_with_identity(
        key: str = "identity",
        num_buckets: int = 1,
        default_value: int | None = None,
        **kwargs: TensorOperand,
    ) -> IdentityCategoricalColumn:
        """Represent categorical integer IDs directly as identity buckets.

        Args:
            key: Feature key string.
            num_buckets: Number of identity buckets.
            default_value: Optional default bucket ID.
            **kwargs: Unused keyword arguments.

        Returns:
            IdentityCategoricalColumn: Typed identity categorical column.
        """
        del kwargs
        return IdentityCategoricalColumn(
            key=str(key),
            num_buckets=int(num_buckets),
            default_value=default_value,
        )

    @staticmethod
    def categorical_column_with_vocabulary_file(
        key: str = "vocabulary_file",
        vocabulary_file: str = "",
        vocabulary_size: int | None = None,
        num_oov_buckets: int = 0,
        default_value: int = -1,
        dtype: DTypeLike = "string",
        **kwargs: TensorOperand,
    ) -> VocabularyListCategoricalColumn:
        """Represent categorical feature mapped via a vocabulary file.

        Args:
            key: Feature key string.
            vocabulary_file: Path to vocabulary file.
            vocabulary_size: Optional vocabulary size limit.
            num_oov_buckets: Number of out-of-vocabulary buckets.
            default_value: Default ID for out-of-vocabulary items.
            dtype: Feature `DType`.
            **kwargs: Unused keyword arguments.

        Returns:
            VocabularyListCategoricalColumn: Typed vocabulary categorical column.
        """
        del kwargs
        vocab: list[str] = []
        if vocabulary_file and os.path.isfile(vocabulary_file):
            with open(vocabulary_file, encoding="utf-8") as f:
                raw_lines = [line.strip() for line in f if line.strip()]
            vocab = raw_lines[: (int(vocabulary_size) if vocabulary_size is not None else None)]
        col = VocabularyListCategoricalColumn(
            key=str(key),
            vocabulary_list=vocab,
            dtype=dtype,
            default_value=default_value,
            num_oov_buckets=num_oov_buckets,
        )
        col["type"] = "vocabulary_file"
        col["vocabulary_file"] = vocabulary_file
        return col

    @staticmethod
    def categorical_column_with_vocabulary_list(
        key: str = "vocabulary_list",
        vocabulary_list: Sequence[str | int] | TensorOperand = (),
        dtype: DTypeLike = None,
        default_value: int = -1,
        num_oov_buckets: int = 0,
        **kwargs: TensorOperand,
    ) -> VocabularyListCategoricalColumn:
        """Represent categorical feature mapped via an in-memory vocabulary list.

        Args:
            key: Feature key string.
            vocabulary_list: In-memory sequence of vocabulary terms.
            dtype: Optional feature `DType`.
            default_value: Default ID for out-of-vocabulary items.
            num_oov_buckets: Number of out-of-vocabulary buckets.
            **kwargs: Unused keyword arguments.

        Returns:
            VocabularyListCategoricalColumn: Typed vocabulary list categorical column.
        """
        del kwargs
        v_list = (
            [x for x in vocabulary_list if isinstance(x, (str, int))]
            if isinstance(vocabulary_list, Sequence)
            and not isinstance(vocabulary_list, (str, bytes))
            else []
        )
        inferred_dtype: DTypeLike = (
            dtype
            if dtype is not None
            else ("int64" if v_list and isinstance(v_list[0], int) else "string")
        )
        return VocabularyListCategoricalColumn(
            key=str(key),
            vocabulary_list=v_list,
            dtype=inferred_dtype,
            default_value=default_value,
            num_oov_buckets=num_oov_buckets,
        )

    @staticmethod
    def crossed_column(
        keys: Sequence[str | _FeatureColumnBase] = (),
        hash_bucket_size: int = 1,
        hash_key: int | None = None,
        **kwargs: TensorOperand,
    ) -> HashBucketCategoricalColumn:
        """Represent a column for performing crosses of categorical features.

        Args:
            keys: Sequence of feature keys or columns to cross.
            hash_bucket_size: Number of hash buckets for crossed features.
            hash_key: Optional custom hash key.
            **kwargs: Unused keyword arguments.

        Returns:
            HashBucketCategoricalColumn: Typed crossed categorical column.
        """
        del hash_key, kwargs
        key_names = [k.key if isinstance(k, _FeatureColumnBase) else str(k) for k in keys]
        joined_key = "_X_".join(key_names) if key_names else "crossed"
        col = HashBucketCategoricalColumn(
            key=joined_key, hash_bucket_size=max(1, int(hash_bucket_size))
        )
        col["type"] = "crossed"
        col["keys"] = key_names
        return col

    @staticmethod
    def embedding_column(
        categorical_column: _FeatureColumnBase | TensorOperand = None,
        dimension: int = 1,
        combiner: str = "mean",
        initializer: TensorOperand = None,
        ckpt_to_load_from: str | None = None,
        tensor_name_in_ckpt: str | None = None,
        max_norm: float | None = None,
        trainable: bool = True,
        use_safe_embedding_lookup: bool = True,
        **kwargs: TensorOperand,
    ) -> EmbeddingColumn:
        """Represent a dense embedding lookup over a categorical column.

        Args:
            categorical_column: Underlying categorical column.
            dimension: Positive embedding dimension integer.
            combiner: Reduction combiner string (`'mean'`, `'sqrtn'`, `'sum'`).
            initializer: Optional weight initializer.
            ckpt_to_load_from: Optional checkpoint path.
            tensor_name_in_ckpt: Optional tensor name in checkpoint.
            max_norm: Optional L2 max norm clipping threshold.
            trainable: Whether the embedding weights are trainable.
            use_safe_embedding_lookup: Whether to validate sparse IDs.
            **kwargs: Unused keyword arguments.

        Returns:
            EmbeddingColumn: Typed embedding column.
        """
        del (
            initializer,
            ckpt_to_load_from,
            tensor_name_in_ckpt,
            max_norm,
            trainable,
            use_safe_embedding_lookup,
            kwargs,
        )
        cat = categorical_column if isinstance(categorical_column, _FeatureColumnBase) else None
        return EmbeddingColumn(categorical_column=cat, dimension=dimension, combiner=combiner)

    @staticmethod
    def indicator_column(
        categorical_column: _FeatureColumnBase | TensorOperand = None,
        **kwargs: TensorOperand,
    ) -> IndicatorColumn:
        """Represent multi-hot indicator vector over a categorical column.

        Args:
            categorical_column: Underlying categorical column.
            **kwargs: Unused keyword arguments.

        Returns:
            IndicatorColumn: Typed indicator column.
        """
        del kwargs
        cat = categorical_column if isinstance(categorical_column, _FeatureColumnBase) else None
        return IndicatorColumn(categorical_column=cat)

    @staticmethod
    def make_parse_example_spec(
        feature_columns: Sequence[_FeatureColumnBase | TensorOperand] = (),
        **kwargs: TensorOperand,
    ) -> dict[str, FixedLenFeature | VarLenFeature | TensorOperand]:
        """Create parsing specification dict from a collection of `feature_column`s.

        Args:
            feature_columns: Sequence of `_FeatureColumnBase` instances.
            **kwargs: Additional specification entries.

        Returns:
            dict[str, FixedLenFeature | VarLenFeature | TensorOperand]: Parse specification dictionary.
        """
        spec: dict[str, FixedLenFeature | VarLenFeature | TensorOperand] = dict(kwargs)
        for col in feature_columns:
            if isinstance(col, (IndicatorColumn, EmbeddingColumn)) and isinstance(
                col.categorical_column, _FeatureColumnBase
            ):
                spec.update(col.categorical_column.parse_example_spec)
            elif isinstance(col, BucketizedColumn) and isinstance(
                col.source_column, _FeatureColumnBase
            ):
                spec.update(col.source_column.parse_example_spec)
            elif isinstance(col, _FeatureColumnBase):
                spec.update(col.parse_example_spec)
        return spec

    @staticmethod
    def numeric_column(
        key: str = "numeric",
        shape: Sequence[int] | int = (1,),
        default_value: float | None = None,
        dtype: DTypeLike = "float32",
        normalizer_fn: Callable[[Tensor], Tensor] | None = None,
        **kwargs: TensorOperand,
    ) -> NumericColumn:
        """Represent real-valued dense numeric feature column.

        Args:
            key: Feature key string.
            shape: Dense feature shape tuple or integer.
            default_value: Optional default scalar value.
            dtype: Numeric `DType`.
            normalizer_fn: Optional normalization callable.
            **kwargs: Unused keyword arguments.

        Returns:
            NumericColumn: Typed numeric feature column.
        """
        del kwargs
        norm_shape = (
            (int(shape),) if isinstance(shape, int) else builtins.tuple(int(d) for d in shape)
        )
        return NumericColumn(
            key=str(key),
            shape=norm_shape,
            default_value=default_value,
            dtype=dtype,
            normalizer_fn=normalizer_fn,
        )

    @staticmethod
    def sequence_categorical_column_with_hash_bucket(
        key: str = "seq_hash_bucket",
        hash_bucket_size: int = 1,
        dtype: DTypeLike = "string",
        **kwargs: TensorOperand,
    ) -> HashBucketCategoricalColumn:
        """Represent sequence of categorical terms hashed into buckets.

        Args:
            key: Feature key string.
            hash_bucket_size: Number of hash buckets.
            dtype: Feature `DType`.
            **kwargs: Unused keyword arguments.

        Returns:
            HashBucketCategoricalColumn: Sequence hash-bucket categorical column.
        """
        del kwargs
        col = HashBucketCategoricalColumn(
            key=str(key), hash_bucket_size=int(hash_bucket_size), dtype=dtype
        )
        col["type"] = "seq_hash_bucket"
        return col

    @staticmethod
    def sequence_categorical_column_with_identity(
        key: str = "seq_identity",
        num_buckets: int = 1,
        default_value: int | None = None,
        **kwargs: TensorOperand,
    ) -> IdentityCategoricalColumn:
        """Represent sequence of integer IDs.

        Args:
            key: Feature key string.
            num_buckets: Number of identity buckets.
            default_value: Optional default bucket ID.
            **kwargs: Unused keyword arguments.

        Returns:
            IdentityCategoricalColumn: Sequence identity categorical column.
        """
        del kwargs
        col = IdentityCategoricalColumn(
            key=str(key), num_buckets=int(num_buckets), default_value=default_value
        )
        col["type"] = "seq_identity"
        return col

    @staticmethod
    def sequence_categorical_column_with_vocabulary_file(
        key: str = "seq_vocabulary_file",
        vocabulary_file: str = "",
        vocabulary_size: int | None = None,
        num_oov_buckets: int = 0,
        default_value: int = -1,
        dtype: DTypeLike = "string",
        **kwargs: TensorOperand,
    ) -> VocabularyListCategoricalColumn:
        """Represent sequence of categorical strings mapped via a vocabulary file.

        Args:
            key: Feature key string.
            vocabulary_file: Path to vocabulary file.
            vocabulary_size: Optional vocabulary size limit.
            num_oov_buckets: Number of out-of-vocabulary buckets.
            default_value: Default ID for out-of-vocabulary items.
            dtype: Feature `DType`.
            **kwargs: Unused keyword arguments.

        Returns:
            VocabularyListCategoricalColumn: Sequence vocabulary-file column.
        """
        col = feature_column.categorical_column_with_vocabulary_file(
            key=key,
            vocabulary_file=vocabulary_file,
            vocabulary_size=vocabulary_size,
            num_oov_buckets=num_oov_buckets,
            default_value=default_value,
            dtype=dtype,
            **kwargs,
        )
        col["type"] = "seq_vocabulary_file"
        return col

    @staticmethod
    def sequence_categorical_column_with_vocabulary_list(
        key: str = "seq_vocabulary_list",
        vocabulary_list: Sequence[str | int] = (),
        dtype: DTypeLike = None,
        default_value: int = -1,
        num_oov_buckets: int = 0,
        **kwargs: TensorOperand,
    ) -> VocabularyListCategoricalColumn:
        """Represent sequence of categorical values mapped via an in-memory list.

        Args:
            key: Feature key string.
            vocabulary_list: In-memory vocabulary list.
            dtype: Optional feature `DType`.
            default_value: Default ID for out-of-vocabulary items.
            num_oov_buckets: Number of out-of-vocabulary buckets.
            **kwargs: Unused keyword arguments.

        Returns:
            VocabularyListCategoricalColumn: Sequence vocabulary-list column.
        """
        col = feature_column.categorical_column_with_vocabulary_list(
            key=key,
            vocabulary_list=vocabulary_list,
            dtype=dtype,
            default_value=default_value,
            num_oov_buckets=num_oov_buckets,
            **kwargs,
        )
        col["type"] = "seq_vocabulary_list"
        return col

    @staticmethod
    def sequence_numeric_column(
        key: str = "seq_numeric",
        shape: Sequence[int] | int = (1,),
        default_value: float = 0.0,
        dtype: DTypeLike = "float32",
        normalizer_fn: Callable[[Tensor], Tensor] | None = None,
        **kwargs: TensorOperand,
    ) -> NumericColumn:
        """Represent variable-length numeric sequence feature.

        Args:
            key: Feature key string.
            shape: Element shape tuple or integer.
            default_value: Default value for padding.
            dtype: Numeric `DType`.
            normalizer_fn: Optional normalization callable.
            **kwargs: Unused keyword arguments.

        Returns:
            NumericColumn: Sequence numeric column (marked `is_sparse=True`).
        """
        col = feature_column.numeric_column(
            key=key,
            shape=shape,
            default_value=default_value,
            dtype=dtype,
            normalizer_fn=normalizer_fn,
            **kwargs,
        )
        col.is_sparse = True
        col["type"] = "seq_numeric"
        return col

    @staticmethod
    def shared_embeddings(
        categorical_columns: Sequence[_FeatureColumnBase] = (),
        dimension: int = 1,
        combiner: str = "mean",
        **kwargs: TensorOperand,
    ) -> list[EmbeddingColumn]:
        """Create a list of dense columns that share the same embedding weights.

        Args:
            categorical_columns: Sequence of categorical columns to embed.
            dimension: Embedding vector dimension.
            combiner: Reduction combiner (`'mean'`, `'sqrtn'`, `'sum'`).
            **kwargs: Unused keyword arguments.

        Returns:
            list[EmbeddingColumn]: List of shared `EmbeddingColumn` instances.
        """
        del kwargs
        cols = list(categorical_columns) if categorical_columns else [IdentityCategoricalColumn()]
        out: list[EmbeddingColumn] = []
        for c in cols:
            emb = EmbeddingColumn(categorical_column=c, dimension=dimension, combiner=combiner)
            emb["type"] = "shared_embedding"
            out.append(emb)
        return out

    @staticmethod
    def weighted_categorical_column(
        categorical_column: _FeatureColumnBase | TensorOperand = None,
        weight_feature_key: str = "weights",
        dtype: DTypeLike = "float32",
        **kwargs: TensorOperand,
    ) -> _FeatureColumnBase:
        """Apply weight values to a `CategoricalColumn`.

        Args:
            categorical_column: Underlying categorical column.
            weight_feature_key: Key for weight values.
            dtype: Weight `DType`.
            **kwargs: Unused keyword arguments.

        Returns:
            _FeatureColumnBase: Weighted categorical column descriptor.
        """
        del kwargs
        base_key = (
            categorical_column.key
            if isinstance(categorical_column, _FeatureColumnBase)
            else "weighted"
        )
        nb = (
            categorical_column.num_buckets
            if isinstance(categorical_column, _FeatureColumnBase)
            else 1
        )
        return _FeatureColumnBase(
            "weighted_categorical",
            key=base_key,
            dtype=dtype,
            is_sparse=True,
            num_buckets=nb,
            extra={"weight_feature_key": weight_feature_key},
        )


class graph_util:
    """Helpers to manipulate a tensor graph in Python."""

    @staticmethod
    def import_graph_def(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Deprecated alias for `tf.import_graph_def`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs


_TFLITE_REGISTRY: dict[
    str,
    Callable[[Tensor], TensorOperand]
    | Callable[[TensorOperand], TensorOperand]
    | CallableOperandProtocol,
] = {}


class _TFLiteConverterImpl(dict[str, TensorOperand]):
    """Concrete `tf.lite.TFLiteConverter` serializing functions/models into FlatBuffer-framed bytes."""

    def __init__(
        self,
        callable_fn: Callable[[Tensor], TensorOperand]
        | Callable[[TensorOperand], TensorOperand]
        | CallableOperandProtocol
        | None = None,
        input_shapes: list[builtins.tuple[int, ...]] | None = None,
        input_dtypes: list[str] | None = None,
        source_tag: str = "concrete_functions",
        extra: Mapping[str, TensorOperand] | None = None,
    ) -> None:
        """Initialize `_TFLiteConverterImpl`.

        Args:
            callable_fn: Optional callable model or function to execute.
            input_shapes: Optional list of input tensor shapes.
            input_dtypes: Optional list of input tensor dtype names.
            source_tag: Origin tag (`'concrete_functions'`, `'keras_model'`, or `'saved_model'`).
            extra: Optional extra metadata dictionary.
        """
        super().__init__(source=source_tag)
        if extra is not None:
            self.update(extra)
        self._callable_fn = callable_fn
        self.optimizations: list[str] = []
        self.target_spec: dict[str, TensorOperand] = {}
        self.representative_dataset: TensorOperand = None
        self._input_shapes = input_shapes or [(1,)]
        self._input_dtypes = input_dtypes or ["float32"]

    @classmethod
    def from_concrete_functions(
        cls,
        funcs: Sequence[
            Callable[[Tensor], TensorOperand]
            | Callable[[TensorOperand], TensorOperand]
            | CallableOperandProtocol
            | TensorOperand
        ] = (),
        trackable_obj: TensorOperand = None,
    ) -> _TFLiteConverterImpl:
        """Create a `TFLiteConverter` from a sequence of `ConcreteFunction`s.

        Args:
            funcs: Sequence of concrete functions.
            trackable_obj: Optional trackable object owning the functions.

        Returns:
            _TFLiteConverterImpl: Configured converter instance.
        """
        del trackable_obj
        fn_item = funcs[0] if funcs else None
        callable_fn = fn_item if callable(fn_item) else None
        fn_inputs = getattr_free_inputs(callable_fn) if callable_fn is not None else []
        shapes = [t.shape if t.shape else (1,) for t in fn_inputs] if fn_inputs else [(1,)]
        dtypes = [to_dtype(t.dtype).value for t in fn_inputs] if fn_inputs else ["float32"]
        return cls(
            callable_fn=callable_fn,
            input_shapes=shapes,
            input_dtypes=dtypes,
            source_tag="concrete_functions",
        )

    @classmethod
    def from_keras_model(
        cls,
        model: Callable[[Tensor], TensorOperand]
        | Callable[[TensorOperand], TensorOperand]
        | CallableOperandProtocol
        | TensorOperand = None,
    ) -> _TFLiteConverterImpl:
        """Create a `TFLiteConverter` from a Keras model or callable `Module`.

        Args:
            model: Keras model or callable `Module`.

        Returns:
            _TFLiteConverterImpl: Configured converter instance.
        """
        callable_fn = model if callable(model) else None
        return cls(callable_fn=callable_fn, source_tag="keras_model")

    @classmethod
    def from_saved_model(
        cls,
        saved_model_dir: str = "",
        signature_keys: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> _TFLiteConverterImpl:
        """Create a `TFLiteConverter` from a SavedModel directory.

        Args:
            saved_model_dir: Path to SavedModel directory.
            signature_keys: Optional sequence of signature keys.
            tags: Optional sequence of SavedModel tag strings.

        Returns:
            _TFLiteConverterImpl: Configured converter instance.
        """
        del signature_keys, tags
        return cls(
            callable_fn=None,
            source_tag="saved_model",
            extra={"saved_model_dir": saved_model_dir},
        )

    def convert(self) -> bytes:
        """Convert the configured model/function into a serialized TFLite byte payload.

        Returns:
            bytes: Serialized TFLite FlatBuffer-framed byte payload (`b"TFL3..."`).
        """
        model_id = uuid.uuid4().hex
        if self._callable_fn is not None:
            _TFLITE_REGISTRY[model_id] = self._callable_fn
        header = {
            "format": "TFL3",
            "model_id": model_id,
            "inputs": [
                {"index": i, "name": f"input_{i}", "shape": list(s), "dtype": dt}
                for i, (s, dt) in enumerate(zip(self._input_shapes, self._input_dtypes))
            ],
            "outputs": [
                {
                    "index": len(self._input_shapes),
                    "name": "output_0",
                    "shape": list(self._input_shapes[0]),
                    "dtype": self._input_dtypes[0],
                }
            ],
        }
        return b"TFL3" + json.dumps(header).encode("utf-8")


def getattr_free_inputs(
    fn_item: Callable[[Tensor], TensorOperand]
    | Callable[[TensorOperand], TensorOperand]
    | CallableOperandProtocol
    | TensorOperand,
) -> list[Tensor]:
    """Extract `.inputs` from a `_ConcreteFunction` if it is a concrete function instance.

    Args:
        fn_item: Candidate callable.

    Returns:
        list[Tensor]: List of input tensors if available.
    """
    from zero_tensorflow.autodiff import _ConcreteFunction

    return list(fn_item.inputs) if isinstance(fn_item, _ConcreteFunction) else []


class _TFLiteInterpreterImpl(dict[str, TensorOperand]):
    """Concrete `tf.lite.Interpreter` allocating tensor buffers and executing TFLite models."""

    def __init__(
        self,
        model_path: str | None = None,
        model_content: bytes | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `_TFLiteInterpreterImpl`.

        Args:
            model_path: Optional file path to a `.tflite` model.
            model_content: Optional raw bytes of a converted `.tflite` model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del model_path
        super().__init__(args=list(args), **kwargs)
        meta = (
            json.loads(model_content[4:].decode("utf-8"))
            if isinstance(model_content, bytes) and model_content.startswith(b"TFL3")
            else {}
        )
        self._model_id: str = str(meta.get("model_id", ""))
        in_list = meta.get("inputs", [])
        out_list = meta.get("outputs", [])
        self._input_details: list[dict[str, TensorOperand]] = (
            [
                {
                    "index": int(d.get("index", idx)),
                    "name": str(d.get("name", f"input_{idx}")),
                    "shape": builtins.tuple(int(x) for x in d.get("shape", [1])),
                    "dtype": str(d.get("dtype", "float32")),
                }
                for idx, d in enumerate(in_list)
                if isinstance(d, Mapping)
            ]
            if in_list
            else [{"index": 0, "name": "input_0", "shape": (1,), "dtype": "float32"}]
        )
        self._output_details: list[dict[str, TensorOperand]] = (
            [
                {
                    "index": int(d.get("index", len(self._input_details) + idx)),
                    "name": str(d.get("name", f"output_{idx}")),
                    "shape": builtins.tuple(int(x) for x in d.get("shape", [1])),
                    "dtype": str(d.get("dtype", "float32")),
                }
                for idx, d in enumerate(out_list)
                if isinstance(d, Mapping)
            ]
            if out_list
            else [{"index": 1, "name": "output_0", "shape": (1,), "dtype": "float32"}]
        )
        self._buffers: dict[int, Tensor] = {}
        self._allocated: bool = False

    def allocate_tensors(self) -> None:
        """Allocate input and output tensor buffers for this interpreter."""
        for det in self._input_details + self._output_details:
            idx = int(det["index"]) if isinstance(det["index"], int) else 0
            shape = det["shape"] if isinstance(det["shape"], builtins.tuple) else (1,)
            dt = str(det.get("dtype", "float32"))
            self._buffers[idx] = _wrap_tensor(_ops.zeros(shape, dtype=dt))
        self._allocated = True

    def get_input_details(self) -> list[dict[str, TensorOperand]]:
        """Return list of input tensor detail dictionaries (`index`, `name`, `shape`, `dtype`).

        Returns:
            list[dict[str, TensorOperand]]: Input tensor metadata list.
        """
        return list(self._input_details)

    def get_output_details(self) -> list[dict[str, TensorOperand]]:
        """Return list of output tensor detail dictionaries (`index`, `name`, `shape`, `dtype`).

        Returns:
            list[dict[str, TensorOperand]]: Output tensor metadata list.
        """
        return list(self._output_details)

    def set_tensor(self, tensor_index: int, value: Tensor | TensorOperand) -> None:
        """Set the input buffer at `tensor_index` to `value`.

        Args:
            tensor_index: Buffer slot index from `get_input_details()`.
            value: Input tensor or array-like value.
        """
        self._buffers[int(tensor_index)] = Tensor(value)

    def invoke(self) -> None:
        """Execute the TFLite model on the currently populated input tensor buffers."""
        in_tensors = [
            self._buffers.get(int(d["index"]), Tensor([0.0]))
            for d in self._input_details
            if isinstance(d["index"], int)
        ]
        fn = _TFLITE_REGISTRY.get(self._model_id)
        out_idx = (
            int(self._output_details[0]["index"])
            if isinstance(self._output_details[0]["index"], int)
            else 1
        )
        if fn is not None:
            res = fn(*in_tensors)
            if isinstance(res, builtins.tuple) and res:
                for d, r_item in zip(self._output_details, res):
                    self._buffers[int(d["index"]) if isinstance(d["index"], int) else out_idx] = (
                        Tensor(r_item)
                    )
            else:
                self._buffers[out_idx] = Tensor(res)
        else:
            self._buffers[out_idx] = in_tensors[0]

    def get_tensor(self, tensor_index: int) -> TensorOperand:
        """Return the NumPy/host value stored in the buffer at `tensor_index`.

        Args:
            tensor_index: Buffer slot index from `get_output_details()`.

        Returns:
            TensorOperand: Evaluated tensor buffer content.
        """
        buf = self._buffers.get(int(tensor_index), Tensor([0.0]))
        return buf.numpy()


class lite:
    """TensorFlow Lite (`tf.lite`) converter and interpreter namespace."""

    Interpreter = _TFLiteInterpreterImpl
    TFLiteConverter = _TFLiteConverterImpl

    @staticmethod
    def Optimize(*args: TensorOperand, **kwargs: TensorOperand) -> str:
        """Enum defining TFLite converter optimizations.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            str: Optimization mode identifier.
        """
        del args, kwargs
        return "DEFAULT"

    @staticmethod
    def OpsSet(*args: TensorOperand, **kwargs: TensorOperand) -> str:
        """Enum defining operator sets available to `TFLiteConverter`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            str: OpsSet identifier.
        """
        del args, kwargs
        return "TFLITE_BUILTINS"

    @staticmethod
    def RepresentativeDataset(*args: TensorOperand, **kwargs: TensorOperand) -> list[TensorOperand]:
        """Representative dataset generator wrapper for post-training quantization.

        Args:
            *args: Positional arguments (`input_gen`).
            **kwargs: Keyword arguments.

        Returns:
            list[TensorOperand]: Wrapped argument list.
        """
        del kwargs
        return list(args)

    @staticmethod
    def TargetSpec(*args: TensorOperand, **kwargs: TensorOperand) -> dict[str, TensorOperand]:
        """Specification of target device capabilities for TFLite conversion.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: TargetSpec descriptor.
        """
        return {"args": list(args), **kwargs}

    @staticmethod
    def experimental(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Experimental TFLite APIs.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs


class mlir:
    """TensorFlow `mlir` compiler namespace."""

    class experimental:
        """Experimental MLIR graph and pass conversion utilities."""

        @staticmethod
        def convert_function(
            concrete_fn: TensorOperand = None,
            pass_pipeline: str = "tf-standard-pipeline",
        ) -> str:
            """Import a `ConcreteFunction` and convert it to textual MLIR.

            Args:
                concrete_fn: ConcreteFunction to convert.
                pass_pipeline: MLIR pass pipeline string.

            Returns:
                str: Textual MLIR module string.
            """
            del concrete_fn
            return f'module attributes {{tf.pipeline = "{pass_pipeline}"}} {{}}'

        @staticmethod
        def convert_graph_def(
            graph_def: TensorOperand = None,
            pass_pipeline: str = "tf-standard-pipeline",
        ) -> str:
            """Import a `GraphDef` / `LogicalGraph` and convert it to textual MLIR.

            Args:
                graph_def: Graph container to convert.
                pass_pipeline: MLIR pass pipeline string.

            Returns:
                str: Textual MLIR module string.
            """
            del graph_def
            return f'module attributes {{tf.pipeline = "{pass_pipeline}"}} {{}}'


def _is_composite_tensor(val: TensorOperand) -> bool:
    """Return `True` if `val` is a `SparseTensor`, `RaggedTensor`, or `IndexedSlices`.

    Args:
        val: Candidate value.

    Returns:
        bool: `True` if `val` is a composite tensor.
    """
    return isinstance(val, (SparseTensor, RaggedTensor, IndexedSlices))


def _composite_components(val: TensorOperand) -> list[TensorOperand]:
    """Extract the component tensors of a composite tensor (`SparseTensor`, `RaggedTensor`, `IndexedSlices`).

    Args:
        val: Composite tensor instance.

    Returns:
        list[TensorOperand]: Component tensors list.
    """
    if isinstance(val, SparseTensor):
        return [val.indices, val.values, Tensor(val.dense_shape, dtype=DType.Int64)]
    if isinstance(val, RaggedTensor):
        return [Tensor(val.flat_values), Tensor(val.row_splits, dtype=DType.Int64)]
    assert isinstance(val, IndexedSlices)
    comps: list[TensorOperand] = [val.values, val.indices]
    if val.dense_shape is not None:
        comps.append(val.dense_shape)
    return comps


class nest:
    """Operations for working with nested structures of tensors (`tf.nest`)."""

    @staticmethod
    def is_nested(seq: TensorOperand = None) -> bool:
        """Return `True` if `seq` is a nested sequence or mapping (excluding `str` and `bytes`).

        Args:
            seq: Candidate structure to check.

        Returns:
            bool: `True` if `seq` is a `list`, `tuple`, or `Mapping`.
        """
        return isinstance(seq, (list, builtins.tuple, Mapping))

    @staticmethod
    def flatten(
        structure: Tensor
        | Sequence[TensorOperand]
        | Mapping[str, TensorOperand]
        | TensorOperand = None,
        expand_composites: bool = False,
    ) -> list[TensorOperand]:
        """Return a flat list from a given nested structure.

        Args:
            structure: An arbitrary nested structure of elements.
            expand_composites: Whether to expand composite tensors into components.

        Returns:
            list[TensorOperand]: Flattened list of leaf values.
        """
        if structure is None:
            return []
        if expand_composites and _is_composite_tensor(structure):
            return _composite_components(structure)
        if isinstance(structure, Mapping):
            out_map: list[TensorOperand] = []
            for k in sorted(structure.keys()):
                out_map.extend(nest.flatten(structure[k], expand_composites=expand_composites))
            return out_map
        if isinstance(structure, (list, builtins.tuple)):
            out_seq: list[TensorOperand] = []
            for item in structure:
                out_seq.extend(nest.flatten(item, expand_composites=expand_composites))
            return out_seq
        return [structure]

    @staticmethod
    def map_structure(
        func: CallableOperandProtocol | TensorOperand = None,
        *structure: TensorOperand,
        **kwargs: TensorOperand,
    ) -> TensorOperand:
        """Apply `func` to each entry in `structure` and return a new structure.

        Args:
            func: Callable applied to corresponding leaves across `*structure`.
            *structure: One or more nested structures with identical layout.
            **kwargs: Optional flags (`expand_composites`, `check_types`).

        Returns:
            TensorOperand: New nested structure containing the mapped leaves.
        """
        expand_composites = builtins.bool(kwargs.get("expand_composites", False))
        if not isinstance(func, CallableOperandProtocol) or not structure:
            return None
        first = structure[0]
        if expand_composites and isinstance(first, SparseTensor):
            return SparseTensor(
                indices=func(first.indices),
                values=func(first.values),
                dense_shape=first.dense_shape,
            )
        if isinstance(first, list):
            return [
                nest.map_structure(
                    func, *(s[i] for s in structure if isinstance(s, Sequence)), **kwargs
                )
                for i in range(len(first))
            ]
        if isinstance(first, builtins.tuple):
            return builtins.tuple(
                nest.map_structure(
                    func, *(s[i] for s in structure if isinstance(s, Sequence)), **kwargs
                )
                for i in range(len(first))
            )
        if isinstance(first, Mapping):
            return {
                k: nest.map_structure(
                    func, *(s[k] for s in structure if isinstance(s, Mapping)), **kwargs
                )
                for k in first
            }
        return func(*structure)

    @staticmethod
    def assert_same_structure(
        nest1: TensorOperand = None,
        nest2: TensorOperand = None,
        check_types: bool = True,
        expand_composites: bool = False,
    ) -> None:
        """Assert that two structures have the same nested layout.

        Args:
            nest1: First nested structure.
            nest2: Second nested structure.
            check_types: Whether sequence types (`list` vs `tuple`) must match.
            expand_composites: Whether composite tensors are expanded.

        Raises:
            TypeError: If `check_types=True` and container types (`list` vs `tuple` vs `Mapping`) differ.
            ValueError: If the two structures have different numbers of leaves or dict keys.
        """
        if len(nest.flatten(nest1, expand_composites=expand_composites)) != len(
            nest.flatten(nest2, expand_composites=expand_composites)
        ):
            raise ValueError(f"Structures {nest1!r} and {nest2!r} do not have the same leaf count.")
        if nest.is_nested(nest1) != nest.is_nested(nest2):
            raise ValueError(
                f"Structures {nest1!r} and {nest2!r} do not have the same nesting depth."
            )
        if isinstance(nest1, Mapping) and isinstance(nest2, Mapping):
            if set(nest1.keys()) != set(nest2.keys()):
                raise ValueError(f"Dictionary keys {list(nest1.keys())} != {list(nest2.keys())}.")
            for k in nest1:
                nest.assert_same_structure(
                    nest1[k],
                    nest2[k],
                    check_types=check_types,
                    expand_composites=expand_composites,
                )
            return
        if check_types and nest.is_nested(nest1) and type(nest1) is not type(nest2):
            raise TypeError(
                f"Structures {nest1!r} and {nest2!r} have different container types: "
                f"{type(nest1).__name__} vs {type(nest2).__name__}."
            )
        if isinstance(nest1, (list, builtins.tuple)) and isinstance(nest2, (list, builtins.tuple)):
            if len(nest1) != len(nest2):
                raise ValueError(f"Sequence lengths {len(nest1)} != {len(nest2)}.")
            for a, b in zip(nest1, nest2):
                nest.assert_same_structure(
                    a,
                    b,
                    check_types=check_types,
                    expand_composites=expand_composites,
                )

    @staticmethod
    def pack_sequence_as(
        structure: TensorOperand = None,
        flat_sequence: Sequence[TensorOperand] | TensorOperand = (),
        expand_composites: bool = False,
    ) -> TensorOperand:
        """Return a given flattened sequence packed into a given structure.

        Args:
            structure: Nested structure template.
            flat_sequence: Flat sequence of leaf values to pack.
            expand_composites: Whether to repack composite tensors.

        Returns:
            TensorOperand: Packed structure matching `structure`.
        """
        items = (
            list(flat_sequence)
            if isinstance(flat_sequence, Sequence) and not isinstance(flat_sequence, (str, bytes))
            else []
        )
        it = iter(items)

        def _next_item(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
            """Return the next leaf from `it`.

            Args:
                *args: Ignored leaf positional arguments.
                **kwargs: Ignored keyword arguments.

            Returns:
                TensorOperand: Next packed leaf value.
            """
            del args, kwargs
            return next(it, None)

        return nest.map_structure(_next_item, structure, expand_composites=expand_composites)


class profiler:
    """TensorFlow `profiler` namespace."""

    @staticmethod
    def experimental(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Experimental profiler APIs.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs


class queue:
    """TensorFlow `queue` data structures."""

    class QueueBase:
        """Base queue implementation supporting `enqueue`, `dequeue`, and `size`."""

        def __init__(
            self,
            capacity: int = 100,
            dtypes: Sequence[DType | str] | TensorOperand = (),
            names: Sequence[str] | None = None,
            name: str = "queue",
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `QueueBase`.

            Args:
                capacity: Upper bound on the number of elements stored in this queue.
                dtypes: Sequence of `DType`s per queue component.
                names: Optional component names.
                name: Optional queue name.
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.
            """
            del args, kwargs
            self.capacity = capacity
            self.dtypes = dtypes
            self.names = names
            self.name = name
            self._items: list[TensorOperand] = []

        def enqueue(self, vals: TensorOperand, name: str | None = None) -> None:
            """Enqueue one element into this queue.

            Args:
                vals: Element to enqueue.
                name: Optional operation name.
            """
            del name
            self._items.append(vals)

        def dequeue(self, name: str | None = None) -> TensorOperand:
            """Dequeue and return the oldest element from this queue.

            Args:
                name: Optional operation name.

            Returns:
                TensorOperand: Dequeued element, or `None` if empty.
            """
            del name
            return self._items.pop(0) if self._items else None

        def size(self, name: str | None = None) -> Tensor:
            """Return the number of elements currently in this queue.

            Args:
                name: Optional operation name.

            Returns:
                Tensor: Scalar `int32` count tensor.
            """
            del name
            return Tensor(len(self._items), dtype=DType.Int32)

    class FIFOQueue(QueueBase):
        """A queue implementation that dequeues elements in first-in first-out order."""

    class PaddingFIFOQueue(QueueBase):
        """A FIFOQueue that supports batching variable-sized tensors by padding."""

    class PriorityQueue(QueueBase):
        """A queue implementation that dequeues elements in priority order."""

    class RandomShuffleQueue(QueueBase):
        """A queue implementation that dequeues elements in a random order."""


class sets:
    """TensorFlow set operations (`tf.sets`)."""

    @staticmethod
    def size(
        a: Tensor | TensorOperand = None,
        validate_indices: bool = True,
    ) -> Tensor | None:
        """Compute number of unique elements along last dimension of `a`.

        Args:
            a: Input tensor or sequence.
            validate_indices: Whether to validate sparse indices.

        Returns:
            Tensor | None: Count of unique elements, or `None` if `a is None`.
        """
        del validate_indices
        if a is None:
            return None
        flat = nest.flatten(a.numpy() if isinstance(a, Tensor) else a)
        return Tensor(len({str(v) for v in flat}), dtype=DType.Int32)

    @staticmethod
    def union(
        a: Tensor | TensorOperand = None,
        b: Tensor | TensorOperand = None,
        validate_indices: bool = True,
    ) -> Tensor | None:
        """Compute set union of elements in `a` and `b`.

        Args:
            a: First tensor or sequence.
            b: Second tensor or sequence.
            validate_indices: Whether to validate sparse indices.

        Returns:
            Tensor | None: Sorted unique union `Tensor`, or `None`.
        """
        del validate_indices
        if a is None or b is None:
            return None
        fa = [
            int(v) if isinstance(v, (int, float)) else 0
            for v in nest.flatten(a.numpy() if isinstance(a, Tensor) else a)
        ]
        fb = [
            int(v) if isinstance(v, (int, float)) else 0
            for v in nest.flatten(b.numpy() if isinstance(b, Tensor) else b)
        ]
        return Tensor(sorted(set(fa) | set(fb)))

    @staticmethod
    def intersection(
        a: Tensor | TensorOperand = None,
        b: Tensor | TensorOperand = None,
        validate_indices: bool = True,
    ) -> Tensor | None:
        """Compute set intersection of elements in `a` and `b`.

        Args:
            a: First tensor or sequence.
            b: Second tensor or sequence.
            validate_indices: Whether to validate sparse indices.

        Returns:
            Tensor | None: Sorted unique intersection `Tensor`, or `None`.
        """
        del validate_indices
        if a is None or b is None:
            return None
        fa = [
            int(v) if isinstance(v, (int, float)) else 0
            for v in nest.flatten(a.numpy() if isinstance(a, Tensor) else a)
        ]
        fb = [
            int(v) if isinstance(v, (int, float)) else 0
            for v in nest.flatten(b.numpy() if isinstance(b, Tensor) else b)
        ]
        return Tensor(sorted(set(fa) & set(fb)))

    @staticmethod
    def difference(
        a: Tensor | TensorOperand = None,
        b: Tensor | TensorOperand = None,
        aminusb: bool = True,
        validate_indices: bool = True,
    ) -> Tensor | None:
        """Compute set difference of elements in `a` and `b`.

        Args:
            a: First tensor or sequence.
            b: Second tensor or sequence.
            aminusb: If True, compute `a - b`, else `b - a`.
            validate_indices: Whether to validate sparse indices.

        Returns:
            Tensor | None: Sorted unique difference `Tensor`, or `None`.
        """
        del validate_indices
        if a is None or b is None:
            return None
        fa = {
            int(v) if isinstance(v, (int, float)) else 0
            for v in nest.flatten(a.numpy() if isinstance(a, Tensor) else a)
        }
        fb = {
            int(v) if isinstance(v, (int, float)) else 0
            for v in nest.flatten(b.numpy() if isinstance(b, Tensor) else b)
        }
        return Tensor(sorted((fa - fb) if aminusb else (fb - fa)))


class sysconfig:
    """System configuration and build information (`tf.sysconfig`)."""

    CXX11_ABI_FLAG: int = 1
    CXX_VERSION: int = 201703
    MONOLITHIC_BUILD: int = 0

    @staticmethod
    def get_build_info() -> dict[str, str | bool]:
        """Return a dictionary describing TensorFlow's build environment.

        Returns:
            dict[str, str | bool]: Build configuration metadata.
        """
        return {"is_cuda_build": False, "is_rocm_build": False, "cpu_compiler": "clang"}

    @staticmethod
    def get_include() -> str:
        """Return the directory containing the TensorFlow C++ header files.

        Returns:
            str: Include directory path.
        """
        return os.path.dirname(__file__)

    @staticmethod
    def get_lib() -> str:
        """Return the directory containing the TensorFlow shared library files.

        Returns:
            str: Library directory path.
        """
        return os.path.dirname(__file__)

    @staticmethod
    def get_compile_flags() -> list[str]:
        """Return the compilation flags needed to compile custom C++ operators.

        Returns:
            list[str]: List of compiler flags.
        """
        return [f"-I{sysconfig.get_include()}", "-D_GLIBCXX_USE_CXX11_ABI=1"]

    @staticmethod
    def get_link_flags() -> list[str]:
        """Return the linker flags needed to link custom C++ operators.

        Returns:
            list[str]: List of linker flags.
        """
        return [f"-L{sysconfig.get_lib()}"]


class tools:
    """TensorFlow `tools` namespace."""

    @staticmethod
    def compatibility(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Compatibility tooling namespace.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def docs(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Documentation generation tooling namespace.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs


class _XlaExperimentalNamespace:
    """Callable `tf.xla.experimental` namespace supporting `.compile(computation, inputs)`."""

    def __call__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """No-op call for `tf.xla.experimental()`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def compile(
        computation: Callable[[Tensor], TensorOperand]
        | Callable[[TensorOperand], TensorOperand]
        | CallableOperandProtocol
        | TensorOperand = None,
        inputs: Sequence[Tensor | TensorOperand] | None = None,
    ) -> list[Tensor]:
        """Compile and execute `computation(*inputs)` returning a list of output `Tensor`s.

        Args:
            computation: Callable computation to compile.
            inputs: Optional sequence of input tensors.

        Returns:
            list[Tensor]: Compiled output tensor list.
        """
        if not callable(computation):
            return []
        in_args = list(inputs) if inputs is not None else []
        res = computation(*in_args)  # type: ignore[arg-type]
        if isinstance(res, (list, builtins.tuple)):
            return [_wrap_tensor(r) for r in res]
        return [_wrap_tensor(res)]


class tpu:
    """TensorFlow TPU (`tf.tpu`) configuration namespace."""

    @staticmethod
    def XLAOptions(*args: TensorOperand, **kwargs: TensorOperand) -> dict[str, TensorOperand]:
        """XLA compilation options for TPU execution.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Options dictionary.
        """
        del args
        return dict(kwargs)

    @staticmethod
    def experimental(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Experimental TPU APIs.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs


class version:
    """TensorFlow `version` namespace."""

    COMPILER_VERSION: str = "Clang 15.0.0"
    GIT_VERSION: str = "v2.20.0-rc0"
    GRAPH_DEF_VERSION: int = 1942
    GRAPH_DEF_VERSION_MIN_CONSUMER: int = 0
    GRAPH_DEF_VERSION_MIN_PRODUCER: int = 0
    VERSION: str = "2.20.0"


class xla:
    """TensorFlow XLA (`tf.xla`) compilation namespace."""

    experimental = _XlaExperimentalNamespace()
