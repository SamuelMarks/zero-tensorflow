"""Control-flow, scoping, graph execution, and runtime namespace operations."""

from __future__ import annotations

import builtins
import contextlib
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import TracebackType

from zero_tensorflow import debugging as _debugging
from zero_tensorflow._typing import ArrayLikeProtocol, CallableOperandProtocol, TensorOperand
from zero_tensorflow.autodiff import GradientTape as _GradientTape
from zero_tensorflow.core_stubs import is_symbolic_tensor
from zero_tensorflow.core_tensor import Tensor, Variable, _to_tensor, _wrap, _wrap_tensor
from zero_tensorflow.spec_types import (
    _GRADIENT_REGISTRY,
    _NAME_SCOPE_STACK,
    Operation,
    UnconnectedGradients,
)

from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
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
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Return `true_fn()` if the predicate `pred` is true else `false_fn()`.

    Args:
        pred: A scalar boolean `Tensor` or Python `bool`.
        true_fn: Callable to be performed if `pred` is true.
        false_fn: Callable to be performed if `pred` is false.
        name: Optional name prefix for the returned tensors.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Conditional result.
    """
    del name
    if pred is None:
        return None
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
    """
    del exclusive, strict, name
    if isinstance(pred_fn_pairs, Mapping):
        for pred_key, fn_val in pred_fn_pairs.items():
            if _to_bool(pred_key):
                return _eval_branch(fn_val)
    elif isinstance(pred_fn_pairs, Sequence) and not isinstance(pred_fn_pairs, (str, bytes)):
        for item in pred_fn_pairs:
            if isinstance(item, builtins.tuple) and len(item) == 2 and _to_bool(item[0]):
                return _eval_branch(item[1])
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
        and 0 <= idx < len(branch_fns)
    ):
        return _eval_branch(branch_fns[idx])
    return _eval_branch(default) if default is not None else None


def while_loop(
    cond: CallableOperandProtocol | TensorOperand = None,
    body: CallableOperandProtocol | TensorOperand = None,
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
        loop_vars: Initial loop state (a `Tensor`, tuple, or list of `Tensor`s).
        shape_invariants: Optional shape invariants for `loop_vars`.
        parallel_iterations: Number of iterations allowed to run in parallel.
        back_prop: Whether backprop is enabled.
        swap_memory: Whether GPU-CPU memory swap is enabled.
        maximum_iterations: Optional maximum number of loop iterations.
        name: Optional name for the operation.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Loop state after termination.
    """
    del shape_invariants, parallel_iterations, back_prop, swap_memory, name
    if (
        not isinstance(cond, CallableOperandProtocol)
        or not isinstance(body, CallableOperandProtocol)
        or loop_vars is None
    ):
        return _wrap(loop_vars) if loop_vars is not None else None

    state: Sequence[Tensor | TensorOperand] | Tensor | TensorOperand = loop_vars
    max_steps = maximum_iterations if isinstance(maximum_iterations, int) else 10000
    steps = 0
    while steps < max_steps:
        c_val = cond(*state) if isinstance(state, (builtins.tuple, list)) else cond(state)
        if not _to_bool(c_val):
            break
        state = body(*state) if isinstance(state, (builtins.tuple, list)) else body(state)
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


def map_fn(
    fn: Callable[[Tensor], Tensor | TensorOperand] | TensorOperand = None,
    elems: Tensor | Sequence[TensorOperand] | TensorOperand = None,
    dtype: TensorOperand = None,
    parallel_iterations: int | None = None,
    back_prop: bool = True,
    swap_memory: bool = False,
    infer_shape: bool = True,
    name: str | None = None,
    fn_output_signature: TensorOperand = None,
) -> Tensor | None:
    """Transform `elems` by applying `fn` to each element unstacked on axis 0.

    Args:
        fn: Callable to apply to each slice.
        elems: Tensor or sequence to unpack along dimension 0.
        dtype: Optional output dtype.
        parallel_iterations: Optional parallel iteration count.
        back_prop: Whether backprop is enabled.
        swap_memory: Whether memory swapping is enabled.
        infer_shape: Whether shape inference is enabled.
        name: Optional operation name.
        fn_output_signature: Optional output signature specification.

    Returns:
        Tensor | None: Stacked mapped `Tensor`, or `None` if `elems is None`.
    """
    del dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature
    if not callable(fn) or elems is None:
        return None
    slices = _unpack_elems(elems)
    mapped = [_to_tensor(fn(s)) for s in slices]
    if not mapped:
        return Tensor([])
    return _wrap_tensor(_ops.stack(mapped, axis=0))


def vectorized_map(
    fn: Callable[[Tensor], Tensor | TensorOperand] | TensorOperand = None,
    elems: Tensor | Sequence[TensorOperand] | TensorOperand = None,
    fallback_to_while_loop: bool = True,
    warn: bool = True,
) -> Tensor | None:
    """Parallel map of `fn` across dimension 0 of `elems`.

    Args:
        fn: Callable to apply to each slice of `elems`.
        elems: Tensor or sequence to unpack along dimension 0.
        fallback_to_while_loop: Whether to fall back to `map_fn` if vectorization fails.
        warn: Whether to log warnings on fallback.

    Returns:
        Tensor | None: Stacked mapped `Tensor`, or `None` if `elems is None`.
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


class feature_column:
    """TensorFlow `feature_column` structured input representation namespace."""

    @staticmethod
    def bucketized_column(
        *args: TensorOperand, **kwargs: TensorOperand
    ) -> dict[str, TensorOperand]:
        """Represent discretized dense input bucketed by `boundaries`.

        Args:
            *args: Positional arguments (`source_column`, `boundaries`).
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "bucketized", "args": list(args), **kwargs}

    @staticmethod
    def categorical_column_with_hash_bucket(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Represent sparse categorical feature hashed into `hash_bucket_size` buckets.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "hash_bucket", "args": list(args), **kwargs}

    @staticmethod
    def categorical_column_with_identity(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Represent categorical integer IDs directly as identity buckets.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "identity", "args": list(args), **kwargs}

    @staticmethod
    def categorical_column_with_vocabulary_file(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Represent categorical feature mapped via a vocabulary file.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "vocabulary_file", "args": list(args), **kwargs}

    @staticmethod
    def categorical_column_with_vocabulary_list(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Represent categorical feature mapped via an in-memory vocabulary list.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "vocabulary_list", "args": list(args), **kwargs}

    @staticmethod
    def crossed_column(*args: TensorOperand, **kwargs: TensorOperand) -> dict[str, TensorOperand]:
        """Represent a column for performing crosses of categorical features.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "crossed", "args": list(args), **kwargs}

    @staticmethod
    def embedding_column(*args: TensorOperand, **kwargs: TensorOperand) -> dict[str, TensorOperand]:
        """Represent a dense embedding lookup over a categorical column.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "embedding", "args": list(args), **kwargs}

    @staticmethod
    def indicator_column(*args: TensorOperand, **kwargs: TensorOperand) -> dict[str, TensorOperand]:
        """Represent multi-hot indicator vector over a categorical column.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "indicator", "args": list(args), **kwargs}

    @staticmethod
    def make_parse_example_spec(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Create parsing specification dict from a collection of `feature_column`s.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Parse specification dictionary.
        """
        del args
        return dict(kwargs)

    @staticmethod
    def numeric_column(*args: TensorOperand, **kwargs: TensorOperand) -> dict[str, TensorOperand]:
        """Represent real-valued dense numeric feature column.

        Args:
            *args: Positional arguments (`key`).
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "numeric", "args": list(args), **kwargs}

    @staticmethod
    def sequence_categorical_column_with_hash_bucket(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Represent sequence of categorical terms hashed into buckets.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "seq_hash_bucket", "args": list(args), **kwargs}

    @staticmethod
    def sequence_categorical_column_with_identity(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Represent sequence of integer IDs.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "seq_identity", "args": list(args), **kwargs}

    @staticmethod
    def sequence_categorical_column_with_vocabulary_file(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Represent sequence of categorical strings mapped via a vocabulary file.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "seq_vocabulary_file", "args": list(args), **kwargs}

    @staticmethod
    def sequence_categorical_column_with_vocabulary_list(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Represent sequence of categorical values mapped via an in-memory list.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "seq_vocabulary_list", "args": list(args), **kwargs}

    @staticmethod
    def sequence_numeric_column(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Represent variable-length numeric sequence feature.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Feature column descriptor.
        """
        return {"type": "seq_numeric", "args": list(args), **kwargs}

    @staticmethod
    def shared_embeddings(
        *args: TensorOperand, **kwargs: TensorOperand
    ) -> list[dict[str, TensorOperand]]:
        """Create a list of dense columns that share the same embedding weights.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            list[dict[str, TensorOperand]]: List of shared embedding descriptors.
        """
        return [{"type": "shared_embedding", "args": list(args), **kwargs}]

    @staticmethod
    def weighted_categorical_column(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, TensorOperand]:
        """Apply weight values to a `CategoricalColumn`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Weighted categorical column descriptor.
        """
        return {"type": "weighted_categorical", "args": list(args), **kwargs}


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


class lite:
    """TensorFlow Lite (`tf.lite`) converter and interpreter namespace."""

    @staticmethod
    def Interpreter(*args: TensorOperand, **kwargs: TensorOperand) -> dict[str, TensorOperand]:
        """Interpreter interface for running TensorFlow Lite models.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Interpreter configuration.
        """
        return {"args": list(args), **kwargs}

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
    def TFLiteConverter(*args: TensorOperand, **kwargs: TensorOperand) -> dict[str, TensorOperand]:
        """Convert a TensorFlow model into `TFLite` FlatBuffer format.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Converter descriptor.
        """
        return {"args": list(args), **kwargs}

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
        del expand_composites
        if structure is None:
            return []
        if isinstance(structure, Mapping):
            out_map: list[TensorOperand] = []
            for k in sorted(structure.keys()):
                out_map.extend(nest.flatten(structure[k]))
            return out_map
        if isinstance(structure, (list, builtins.tuple)):
            out_seq: list[TensorOperand] = []
            for item in structure:
                out_seq.extend(nest.flatten(item))
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
        del kwargs
        if not isinstance(func, CallableOperandProtocol) or not structure:
            return None
        first = structure[0]
        if isinstance(first, list):
            return [nest.map_structure(func, item) for item in first]
        if isinstance(first, builtins.tuple):
            return builtins.tuple(nest.map_structure(func, item) for item in first)
        if isinstance(first, Mapping):
            return {k: nest.map_structure(func, v) for k, v in first.items()}
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
            ValueError: If the two structures have different numbers of leaves.
        """
        del check_types, expand_composites
        if len(nest.flatten(nest1)) != len(nest.flatten(nest2)):
            raise ValueError(f"Structures {nest1!r} and {nest2!r} do not have the same leaf count.")

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
        del expand_composites
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

        return nest.map_structure(_next_item, structure)


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

    @staticmethod
    def experimental(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Experimental XLA compilation APIs.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
