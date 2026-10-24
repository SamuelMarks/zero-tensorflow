"""Core `Tensor`, `Variable`, dtype conversion, and wrapping primitives."""

from __future__ import annotations

import builtins
import uuid
from collections.abc import Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    DTypeLike,
    IntTuple,
    NestedTensorData,
    TensorOperand,
)

import ml_switcheroo_compiler
from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_ir import AttributeValue, LogicalNode
from typing_extensions import TypeGuard

from zero_keras import ops as _ops  # isort: skip

_DTYPE_ALIASES: dict[str, DType] = {
    "float": DType.Float32,
    "float16": DType.Float16,
    "half": DType.Float16,
    "float32": DType.Float32,
    "float64": DType.Float64,
    "double": DType.Float64,
    "bfloat16": DType.BFloat16,
    "complex": DType.Complex64,
    "complex64": DType.Complex64,
    "complex128": DType.Complex128,
    "int": DType.Int32,
    "int8": DType.Int8,
    "int16": DType.Int16,
    "int32": DType.Int32,
    "int64": DType.Int64,
    "uint8": DType.UInt8,
    "uint16": DType.UInt16,
    "uint32": DType.UInt32,
    "uint64": DType.UInt64,
    "bool": DType.Bool,
    "bool_": DType.Bool,
    "string": DType.String,
    "str": DType.String,
    "bytes": DType.String,
}


def to_dtype(x: DTypeLike | type | TensorOperand = None) -> DType:
    """Convert a string, Python type, or `DType` to a compiler `DType`.

    Args:
        x: Input dtype representation (`DType`, string name, Python type, or `None`).

    Returns:
        DType: Corresponding compiler `DType` (defaults to `DType.Float32` when `None`).

    Raises:
        TypeError: If `x` is not a recognized data type representation.
    """
    if x is None:
        return DType.Float32
    if isinstance(x, DType):
        return x
    if isinstance(x, str):
        normalized = x.strip().lower()
        if normalized in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[normalized]
        raise TypeError(f"Cannot convert {x!r} to a TensorFlow DType.")
    if x is builtins.bool:
        return DType.Bool
    if x is int:
        return DType.Int32
    if x is float:
        return DType.Float32
    if x is complex:
        return DType.Complex64
    if x is str or x is bytes:
        return DType.String
    if isinstance(x, type) and x.__name__.lower() in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[x.__name__.lower()]
    raise TypeError(f"Cannot convert {x!r} of type {type(x).__name__} to a TensorFlow DType.")


def _to_attribute_value(val: NestedTensorData | TensorOrScalar) -> AttributeValue:
    """Convert a scalar or nested sequence into an `AttributeValue`.

    Args:
        val: Candidate constant value.

    Returns:
        AttributeValue: Validated IR attribute value.
    """
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        return [_to_attribute_value(item) for item in val]
    if isinstance(val, (int, float, builtins.bool, str)) or val is None:
        return val
    return str(val)


def _extract_constant_value(
    data_obj: TensorOrScalar,
) -> AttributeValue:
    """Extract a serializable constant value from a tensor data payload.

    Args:
        data_obj: Underlying tensor data payload.

    Returns:
        AttributeValue: Python list or scalar value compatible with LogicalNode.
    """
    if isinstance(data_obj, ArrayLikeProtocol):
        return _to_attribute_value(data_obj.tolist())
    return _to_attribute_value(data_obj)


def _to_tensor(
    x: Tensor | TensorOperand,
    dtype: DTypeLike | TensorOperand = None,
) -> ml_switcheroo_compiler.Tensor:
    """Convert an input value or `Tensor` into a backend compiler `Tensor`.

    Args:
        x: Input tensor, array, scalar, or sequence.
        dtype: Optional target data type.

    Returns:
        ml_switcheroo_compiler.Tensor: Converted compiler tensor.
    """
    original_tensor: Tensor | None = None
    unwrapped: TensorOperand
    if isinstance(x, Tensor):
        original_tensor = x
        unwrapped = x._tensor
    else:
        unwrapped = x

    if isinstance(unwrapped, ml_switcheroo_compiler.Tensor):
        if _tracer.is_tracing and not isinstance(unwrapped.data, ProxyTensor):
            graph_id = id(_tracer.active_graph)
            traced_map = original_tensor._traced_node_ids if original_tensor is not None else None
            if traced_map is not None and graph_id in traced_map:
                out_id = str(traced_map[graph_id])
                pt = ProxyTensor(
                    id=out_id,
                    shape=unwrapped.shape,
                    dtype=unwrapped.dtype.value,
                )
                return ml_switcheroo_compiler.Tensor(
                    data=pt,
                    config=TensorConfig(
                        shape=unwrapped.shape,
                        dtype=unwrapped.dtype,
                        device=unwrapped.device,
                    ),
                )

            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={"value": _extract_constant_value(unwrapped.data)},
                shape_metadata=unwrapped.shape,
            )
            _tracer.add_node(node)
            if traced_map is not None:
                traced_map[graph_id] = out_id
            pt = ProxyTensor(
                id=out_id,
                shape=unwrapped.shape,
                dtype=unwrapped.dtype.value,
            )
            return ml_switcheroo_compiler.Tensor(
                data=pt,
                config=TensorConfig(
                    shape=unwrapped.shape,
                    dtype=unwrapped.dtype,
                    device=unwrapped.device,
                ),
            )
        return unwrapped

    if isinstance(unwrapped, ProxyTensor):
        try:
            dt = (
                to_dtype(unwrapped.dtype)
                if unwrapped.dtype
                else _compiler_config.default_float_dtype
            )
        except TypeError:
            dt = _compiler_config.default_float_dtype
        return ml_switcheroo_compiler.Tensor(
            data=unwrapped,
            config=TensorConfig(
                shape=unwrapped.shape,
                dtype=dt,
                device=_compiler_config.default_device,
            ),
        )

    dt_opt = to_dtype(dtype) if dtype is not None else None
    res: ml_switcheroo_compiler.Tensor = _ops.array(unwrapped, dtype=dt_opt)
    return res


def _wrap(
    x: Tensor | TensorOperand,
) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
    """Wrap a compiler tensor or sequence of compiler tensors into `Tensor`s.

    Args:
        x: Input compiler tensor, `Tensor`, tuple, or list.

    Returns:
        Tensor | tuple[Tensor, ...] | list[Tensor]: Wrapped tensor(s).
    """
    if isinstance(x, Tensor):
        return x
    if isinstance(x, builtins.tuple):
        return builtins.tuple(_wrap_tensor(i) for i in x)
    if isinstance(x, builtins.list):
        return [_wrap_tensor(i) for i in x]
    return Tensor(x)


def _wrap_tensor(x: Tensor | TensorOperand) -> Tensor:
    """Wrap `x` and guarantee a single `Tensor` return value.

    Args:
        x: Input value or compiler tensor.

    Returns:
        Tensor: Wrapped `Tensor` instance.
    """
    return x if isinstance(x, Tensor) else Tensor(x)


def _is_bool_tensor(val: Tensor) -> TypeGuard[builtins.bool]:
    """Narrow a boolean `Tensor` comparison result for `object.__eq__` compatibility.

    Args:
        val: Boolean `Tensor` returned by an equality or inequality operation.

    Returns:
        bool: Always `True` when `val` is a `Tensor` instance.
    """
    return isinstance(val, Tensor)


def _is_numpy_result(
    val: TensorOrScalar,
) -> TypeGuard[float | int | builtins.bool | complex | builtins.list[float]]:
    """Narrow an eager tensor data buffer for `Tensor.numpy()`.

    Args:
        val: Underlying eager tensor data.

    Returns:
        bool: Always `True` for materialized eager data.
    """
    return isinstance(val, (int, float, builtins.bool, complex, ArrayLikeProtocol)) or val is None


class Tensor:
    """Dual-state Tensor primitive supporting both eager and traced execution."""

    def __init__(
        self,
        value: Tensor | TensorOperand = None,
        dtype: DTypeLike | TensorOperand = None,
        _traced_node: LogicalNode | None = None,
    ) -> None:
        """Initialize a `Tensor`.

        Args:
            value: Initial value, array, or tensor.
            dtype: Optional target data type.
            _traced_node: Optional traced IR `LogicalNode`.
        """
        self._traced_node_ids: dict[int, str] = {}
        self._tensor: ml_switcheroo_compiler.Tensor | None
        if _traced_node is not None:
            pt = ProxyTensor(id=_traced_node.id, shape=())
            self._tensor = ml_switcheroo_compiler.Tensor(
                data=pt,
                config=TensorConfig(
                    shape=(),
                    dtype=_compiler_config.default_float_dtype,
                    device=_compiler_config.default_device,
                ),
            )
        elif isinstance(value, Tensor):
            self._tensor = _to_tensor(value, dtype=dtype) if dtype is not None else value._tensor
        elif value is None:
            self._tensor = None
        else:
            self._tensor = _to_tensor(value, dtype=dtype)

    @property
    def shape(self) -> IntTuple:
        """Return the shape tuple of the tensor.

        Returns:
            tuple[int, ...]: Tensor dimensions.
        """
        return self._tensor.shape if self._tensor is not None else ()

    @property
    def dtype(self) -> DType | None:
        """Return the `DType` of the tensor.

        Returns:
            DType | None: Tensor data type or None if uninitialized.
        """
        return to_dtype(self._tensor.dtype.value) if self._tensor is not None else None

    def numpy(
        self,
    ) -> float | int | builtins.bool | complex | builtins.list[float]:
        """Convert the tensor to a NumPy array or Python constant value.

        Returns:
            float | int | bool | complex | list[float]: Materialized array or constant value.

        Raises:
            ValueError: If called on a non-constant symbolic traced tensor.
        """
        if self._tensor is not None and isinstance(self._tensor.data, ProxyTensor):
            current_graph = _tracer.active_graph
            if (
                current_graph is not None
                and self._tensor.data.id in current_graph.nodes
                and current_graph.nodes[self._tensor.data.id].op_type == "Constant"
            ):
                val = current_graph.nodes[self._tensor.data.id].attributes.get("value")
                if _is_numpy_result(val):
                    return val
            raise ValueError("Cannot call array conversion on a traced tensor")
        data = self._tensor.data if self._tensor is not None else None
        assert _is_numpy_result(data)
        return data

    def __add__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise addition `self + other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Sum tensor.
        """
        return _wrap_tensor(_ops.add(_to_tensor(self), _to_tensor(other)))

    def __sub__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise subtraction `self - other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Difference tensor.
        """
        return _wrap_tensor(_ops.subtract(_to_tensor(self), _to_tensor(other)))

    def __mul__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise multiplication `self * other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Product tensor.
        """
        return _wrap_tensor(_ops.multiply(_to_tensor(self), _to_tensor(other)))

    def __truediv__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise true division `self / other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Quotient tensor.
        """
        return _wrap_tensor(_ops.divide(_to_tensor(self), _to_tensor(other)))

    def __neg__(self) -> Tensor:
        """Compute element-wise negation `-self`.

        Returns:
            Tensor: Negated tensor.
        """
        return _wrap_tensor(_ops.negative(_to_tensor(self)))

    def __pow__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise power `self ** other`.

        Args:
            other: Exponent operand.

        Returns:
            Tensor: Power tensor.
        """
        return _wrap_tensor(_ops.power(_to_tensor(self), _to_tensor(other)))

    def __radd__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise addition `other + self`.

        Args:
            other: Left-hand operand.

        Returns:
            Tensor: Sum tensor.
        """
        return _wrap_tensor(_ops.add(_to_tensor(other), _to_tensor(self)))

    def __rsub__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise subtraction `other - self`.

        Args:
            other: Left-hand operand.

        Returns:
            Tensor: Difference tensor.
        """
        return _wrap_tensor(_ops.subtract(_to_tensor(other), _to_tensor(self)))

    def __rmul__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise multiplication `other * self`.

        Args:
            other: Left-hand operand.

        Returns:
            Tensor: Product tensor.
        """
        return _wrap_tensor(_ops.multiply(_to_tensor(other), _to_tensor(self)))

    def __rtruediv__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise true division `other / self`.

        Args:
            other: Left-hand operand.

        Returns:
            Tensor: Quotient tensor.
        """
        return _wrap_tensor(_ops.divide(_to_tensor(other), _to_tensor(self)))

    def __eq__(self, other: builtins.object) -> builtins.bool:
        """Compute element-wise equality `self == other`.

        Args:
            other: Right-hand operand.

        Returns:
            bool: Boolean equality tensor.
        """
        if (
            not isinstance(
                other,
                (
                    Tensor,
                    ml_switcheroo_compiler.Tensor,
                    ProxyTensor,
                    ArrayLikeProtocol,
                    int,
                    float,
                    builtins.bool,
                    complex,
                    str,
                    bytes,
                    Sequence,
                ),
            )
            and other is not None
        ):
            return False
        res = _wrap_tensor(_ops.equal(_to_tensor(self), _to_tensor(other)))
        assert _is_bool_tensor(res)
        return res

    def __ne__(self, other: builtins.object) -> builtins.bool:
        """Compute element-wise inequality `self != other`.

        Args:
            other: Right-hand operand.

        Returns:
            bool: Boolean inequality tensor.
        """
        if (
            not isinstance(
                other,
                (
                    Tensor,
                    ml_switcheroo_compiler.Tensor,
                    ProxyTensor,
                    ArrayLikeProtocol,
                    int,
                    float,
                    builtins.bool,
                    complex,
                    str,
                    bytes,
                    Sequence,
                ),
            )
            and other is not None
        ):
            return True
        res = _wrap_tensor(_ops.not_equal(_to_tensor(self), _to_tensor(other)))
        assert _is_bool_tensor(res)
        return res

    def __lt__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise less-than `self < other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Boolean comparison tensor.
        """
        return _wrap_tensor(_ops.less(_to_tensor(self), _to_tensor(other)))

    def __le__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise less-than-or-equal `self <= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Boolean comparison tensor.
        """
        return _wrap_tensor(_ops.less_equal(_to_tensor(self), _to_tensor(other)))

    def __gt__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise greater-than `self > other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Boolean comparison tensor.
        """
        return _wrap_tensor(_ops.greater(_to_tensor(self), _to_tensor(other)))

    def __ge__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise greater-than-or-equal `self >= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Boolean comparison tensor.
        """
        return _wrap_tensor(_ops.greater_equal(_to_tensor(self), _to_tensor(other)))

    def __bool__(self) -> builtins.bool:
        """Convert a scalar eager tensor to a Python boolean.

        Returns:
            bool: Truth value of the scalar tensor.

        Raises:
            TypeError: If the tensor is a symbolic traced tensor.
            ValueError: If the tensor contains more than one element.
        """
        if self._tensor is not None and isinstance(self._tensor.data, ProxyTensor):
            raise TypeError(
                "Using a `tf.Tensor` as a Python `bool` is not allowed in Graph execution."
            )
        arr = self.numpy()
        if isinstance(arr, ArrayLikeProtocol) and arr.size == 1:
            return builtins.bool(arr.item())
        raise ValueError("The truth value of an array with more than one element is ambiguous.")

    def __nonzero__(self) -> builtins.bool:
        """Python 2 compatibility alias for `__bool__`.

        Returns:
            bool: Truth value of the scalar tensor.
        """
        return self.__bool__()

    def __len__(self) -> int:
        """Return the length of the first dimension of the tensor.

        Returns:
            int: First dimension size.

        Raises:
            TypeError: If called on a 0-D scalar tensor.
        """
        if len(self.shape) == 0:
            raise TypeError("Scalar tensor has no `len()`")
        return self.shape[0]


class Variable(Tensor):
    """A mutable Tensor whose value can be updated via assign operations."""

    def __init__(
        self,
        initial_value: Tensor | TensorOperand,
        trainable: builtins.bool = True,
        name: str | None = None,
        dtype: DTypeLike | TensorOperand = None,
    ) -> None:
        """Initialize a `Variable`.

        Args:
            initial_value: Initial value for the variable.
            trainable: Whether the variable is trainable.
            name: Optional name for the variable.
            dtype: Optional target data type.
        """
        super().__init__(initial_value, dtype=dtype)
        self.trainable = trainable
        self.name = name or "Variable:0"

    @property
    def value(self) -> Variable:
        """Return the current value of the variable.

        Returns:
            Variable: Self reference.
        """
        return self

    def read_value(self) -> Tensor:
        """Return a `Tensor` snapshot of the current variable value.

        Returns:
            Tensor: Current variable value as a `Tensor`.
        """
        return Tensor(self._tensor)

    def assign(self, value: Tensor | TensorOperand) -> Variable:
        """Assign a new value to this variable.

        Args:
            value: New value to assign.

        Returns:
            Variable: Updated variable (`self`).
        """
        self._tensor = _to_tensor(value)
        return self

    def assign_add(self, delta: Tensor | TensorOperand) -> Variable:
        """Add `delta` to this variable in-place.

        Args:
            delta: Tensor or scalar delta to add.

        Returns:
            Variable: Updated variable (`self`).
        """
        self._tensor = _ops.add(_to_tensor(self), _to_tensor(delta))
        return self

    def assign_sub(self, delta: Tensor | TensorOperand) -> Variable:
        """Subtract `delta` from this variable in-place.

        Args:
            delta: Tensor or scalar delta to subtract.

        Returns:
            Variable: Updated variable (`self`).
        """
        self._tensor = _ops.subtract(_to_tensor(self), _to_tensor(delta))
        return self


class _TracingContext:
    """Context manager tracking active graph tracing state synchronized with `_tracer`."""

    _current_context: _TracingContext | None = None

    def __init__(self) -> None:
        """Initialize a `_TracingContext` instance."""
        self.active: builtins.bool = _tracer.is_tracing

    @classmethod
    def enter(cls) -> _TracingContext:
        """Enter a new `_TracingContext` and set it as current.

        Returns:
            _TracingContext: Activated tracing context.
        """
        ctx = cls()
        ctx.active = True
        cls._current_context = ctx
        return ctx

    @classmethod
    def exit(cls) -> None:
        """Exit the current `_TracingContext` and clear `_current_context`."""
        cls._current_context = None

    @classmethod
    def get(cls) -> _TracingContext | None:
        """Return the currently active `_TracingContext`, if any.

        Returns:
            _TracingContext | None: Current context or None.
        """
        return cls._current_context


__all__ = [
    "Tensor",
    "Variable",
    "_TracingContext",
    "_extract_constant_value",
    "_to_attribute_value",
    "_to_tensor",
    "_wrap",
    "_wrap_tensor",
    "to_dtype",
]
