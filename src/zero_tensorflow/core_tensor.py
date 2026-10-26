"""Core `Tensor`, `Variable`, dtype conversion, and wrapping primitives."""

from __future__ import annotations

import builtins
import math
import uuid
from collections.abc import Iterator, Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    DTypeLike,
    IntTuple,
    NestedTensorData,
    TensorOperand,
)

import ml_switcheroo_compiler
import ml_switcheroo_compiler.ops as _msc_ops
from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_ir import AttributeValue, LogicalNode
from typing_extensions import TypeAlias, TypeGuard

from zero_keras import ops as _ops  # isort: skip

_SliceItem: TypeAlias = "int | slice[int | None, int | None, int | None] | None | Tensor"
_SliceSpec: TypeAlias = "_SliceItem | builtins.tuple[_SliceItem, ...]"

_DTYPE_ALIASES: dict[str, DType] = {
    "float": DType.Float32,
    "float16": DType.Float16,
    "half": DType.Float16,
    "float32": DType.Float32,
    "float64": DType.Float64,
    "double": DType.Float64,
    "bfloat16": DType.BFloat16,
    "float8_e4m3fn": DType.Float8E4M3FN,
    "float8_e5m2": DType.Float8E5M2,
    "complex": DType.Complex64,
    "complex64": DType.Complex64,
    "complex128": DType.Complex128,
    "int": DType.Int32,
    "int4": DType.Int4,
    "int8": DType.Int8,
    "int16": DType.Int16,
    "int32": DType.Int32,
    "int64": DType.Int64,
    "uint4": DType.UInt4,
    "uint8": DType.UInt8,
    "uint16": DType.UInt16,
    "uint32": DType.UInt32,
    "uint64": DType.UInt64,
    "qint8": DType.Int8,
    "quint8": DType.UInt8,
    "qint16": DType.Int16,
    "quint16": DType.UInt16,
    "qint32": DType.Int32,
    "bool": DType.Bool,
    "bool_": DType.Bool,
    "string": DType.String,
    "str": DType.String,
    "bytes": DType.String,
    "resource": DType.Int64,
    "variant": DType.Int64,
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
            if (
                traced_map is not None
                and graph_id in traced_map
                and _tracer.active_graph is not None
                and str(traced_map[graph_id]) in _tracer.active_graph.nodes
            ):
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


def _slice_nested(
    data: NestedTensorData,
    items: Sequence[_SliceItem],
) -> NestedTensorData:
    """Recursively slice nested Python list/scalar data according to `items`.

    Args:
        data: Nested Python list or scalar data.
        items: Sequence of index/slice/None/Ellipsis items.

    Returns:
        NestedTensorData: Sliced nested data.
    """
    if not items:
        return data
    head = items[0]
    tail = items[1:]
    if head is None:
        return [_slice_nested(data, tail)]
    if isinstance(data, list):
        if isinstance(head, slice):
            return [_slice_nested(row, tail) for row in data[head]]
        return _slice_nested(data[head if isinstance(head, int) else 0], tail)
    return data


def _boolean_mask_nested(
    data: NestedTensorData,
    mask: NestedTensorData,
) -> list[NestedTensorData]:
    """Filter `data` elements where corresponding `mask` entries are truthy.

    Args:
        data: Nested Python list or scalar data.
        mask: Nested boolean mask list or scalar.

    Returns:
        list[NestedTensorData]: Filtered elements list.
    """
    if isinstance(data, list) and isinstance(mask, list):
        out: list[NestedTensorData] = []
        for d_item, m_item in zip(data, mask):
            if isinstance(m_item, list):
                out.extend(_boolean_mask_nested(d_item, m_item))
            elif m_item:
                out.append(d_item)
        return out
    return [data] if mask else []


class _TensorRef:
    """Hashable object-identity reference to a `Tensor` or `Variable` (`tf.Tensor.ref()`)."""

    def __init__(self, tensor: Tensor) -> None:
        """Initialize a `_TensorRef` wrapping `tensor`.

        Args:
            tensor: Target `Tensor` or `Variable`.
        """
        self._tensor = tensor

    def deref(self) -> Tensor:
        """Dereference and return the underlying `Tensor`.

        Returns:
            Tensor: Referenced tensor instance.
        """
        return self._tensor

    def __hash__(self) -> int:
        """Return object-identity hash of the underlying `Tensor`.

        Returns:
            int: Identity hash value.
        """
        return id(self._tensor)

    def __eq__(self, other: builtins.object) -> builtins.bool:
        """Compare identity equality with another `_TensorRef`.

        Args:
            other: Candidate `_TensorRef` to compare.

        Returns:
            bool: `True` when `other` references the exact same `Tensor` object.
        """
        return isinstance(other, _TensorRef) and self._tensor is other._tensor


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
    def data(self) -> TensorOrScalar:
        """Return the underlying compiler tensor or raw data payload.

        Returns:
            TensorOrScalar: Underlying compiler tensor or None if uninitialized.
        """
        return self._tensor if self._tensor is not None else None

    @property
    def ndim(self) -> int:
        """Return the rank (number of dimensions) of the tensor.

        Returns:
            int: Number of dimensions.
        """
        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the total number of elements in the tensor.

        Returns:
            int: Product of dimensions (`0` if uninitialized).
        """
        if self._tensor is None:
            return 0
        return math.prod(self.shape) if self.shape else 1

    @property
    def device(self) -> str:
        """Return the canonical placement device string for the tensor.

        Returns:
            str: Device placement string (e.g., `'/job:localhost/replica:0/task:0/device:CPU:0'`).
        """
        raw_dev = str(self._tensor.device) if self._tensor is not None else "cpu"
        return f"/job:localhost/replica:0/task:0/device:{raw_dev.upper()}:0"

    @property
    def backing_device(self) -> str:
        """Return the physical backing device string for the tensor.

        Returns:
            str: Backing device string.
        """
        return self.device

    @property
    def dtype(self) -> DType | None:
        """Return the `DType` of the tensor.

        Returns:
            DType | None: Tensor data type or None if uninitialized.
        """
        return to_dtype(self._tensor.dtype.value) if self._tensor is not None else None

    def consumers(self) -> list[LogicalNode]:
        """Return traced IR `LogicalNode`s that consume this tensor in the active graph.

        Returns:
            list[LogicalNode]: Consuming IR nodes in the active graph.
        """
        graph = _tracer.active_graph
        if graph is None or self._tensor is None:
            return []
        node_id = (
            self._tensor.data.id
            if isinstance(self._tensor.data, ProxyTensor)
            else self._traced_node_ids.get(id(graph))
        )
        return [n for n in graph.nodes.values() if node_id is not None and node_id in n.inputs]

    def ref(self) -> _TensorRef:
        """Return a hashable `_TensorRef` to this tensor for use in sets/dicts.

        Returns:
            _TensorRef: Hashable reference wrapper.
        """
        return _TensorRef(self)

    def experimental_ref(self) -> _TensorRef:
        """Return a hashable `_TensorRef` to this tensor (experimental alias for `ref()`).

        Returns:
            _TensorRef: Hashable reference wrapper.
        """
        return _TensorRef(self)

    def get_shape(self) -> IntTuple:
        """Return the static shape tuple of this tensor.

        Returns:
            tuple[int, ...]: Tensor dimensions.
        """
        return self.shape

    def set_shape(self, shape: IntTuple | Sequence[int | None] | None) -> None:
        """Update the static shape metadata of this tensor.

        Args:
            shape: Target shape tuple or sequence of dimensions, or `None`.
        """
        if shape is not None and self._tensor is not None:
            resolved = builtins.tuple(int(d) if d is not None else 0 for d in shape)
            self._tensor = ml_switcheroo_compiler.Tensor(
                data=self._tensor.data,
                config=TensorConfig(
                    shape=resolved,
                    dtype=self._tensor.dtype,
                    device=self._tensor.device,
                ),
            )

    def eval(
        self,
        feed_dict: TensorOperand = None,
        session: TensorOperand = None,
    ) -> float | int | builtins.bool | complex | builtins.list[float]:
        """Evaluate this tensor and return its materialized value.

        Args:
            feed_dict: Optional dictionary mapping tensors to feed values.
            session: Optional session in which to evaluate the tensor.

        Returns:
            float | int | bool | complex | list[float]: Materialized tensor value.
        """
        del feed_dict, session
        return self.numpy()

    def __call__(self) -> Tensor:
        """Return `self` when invoked as a callable value accessor.

        Returns:
            Tensor: Self tensor instance.
        """
        return self

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

    def __getitem__(self, idx: _SliceSpec) -> Tensor:
        """Index or slice the tensor using integers, slices, `None`, `...`, or boolean `Tensor`s.

        Args:
            idx: Index, slice, boolean mask `Tensor`, or tuple of slice items.

        Returns:
            Tensor: Sliced or indexed `Tensor`.
        """
        if isinstance(idx, Tensor) and idx.dtype == DType.Bool:
            raw_self = self.numpy()
            raw_mask = idx.numpy()
            py_self = raw_self.tolist() if isinstance(raw_self, ArrayLikeProtocol) else raw_self
            py_mask = raw_mask.tolist() if isinstance(raw_mask, ArrayLikeProtocol) else raw_mask
            return Tensor(_boolean_mask_nested(py_self, py_mask), dtype=self.dtype)

        items: builtins.tuple[_SliceItem, ...] = idx if isinstance(idx, builtins.tuple) else (idx,)
        raw = self.numpy()
        py_data = raw.tolist() if isinstance(raw, ArrayLikeProtocol) else raw
        sliced = _slice_nested(py_data, items)
        return Tensor(sliced, dtype=self.dtype)

    def __iter__(self) -> Iterator[Tensor]:
        """Iterate over the first dimension of an eager tensor.

        Yields:
            Tensor: Sub-tensors along axis 0.

        Raises:
            TypeError: If called on a 0-D scalar tensor or a symbolic traced tensor.
        """
        if self._tensor is not None and isinstance(self._tensor.data, ProxyTensor):
            raise TypeError("Iterating over a symbolic `tf.Tensor` is not allowed.")
        n = len(self)
        for i in range(n):
            yield self[i]

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

    def __matmul__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute matrix multiplication `self @ other`.

        Args:
            other: Right-hand matrix/tensor operand.

        Returns:
            Tensor: Matrix product tensor.
        """
        return _wrap_tensor(_ops.matmul(_to_tensor(self), _to_tensor(other)))

    def __rmatmul__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected matrix multiplication `other @ self`.

        Args:
            other: Left-hand matrix/tensor operand.

        Returns:
            Tensor: Matrix product tensor.
        """
        return _wrap_tensor(_ops.matmul(_to_tensor(other), _to_tensor(self)))

    def __truediv__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise true division `self / other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Quotient tensor.
        """
        return _wrap_tensor(_ops.divide(_to_tensor(self), _to_tensor(other)))

    def __floordiv__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise floor division `self // other`.

        Args:
            other: Right-hand divisor operand.

        Returns:
            Tensor: Floor quotient tensor.
        """
        return _wrap_tensor(_ops.floor_divide(_to_tensor(self), _to_tensor(other)))

    def __rfloordiv__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise floor division `other // self`.

        Args:
            other: Left-hand dividend operand.

        Returns:
            Tensor: Floor quotient tensor.
        """
        return _wrap_tensor(_ops.floor_divide(_to_tensor(other), _to_tensor(self)))

    def __mod__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise remainder `self % other`.

        Args:
            other: Right-hand divisor operand.

        Returns:
            Tensor: Remainder tensor.
        """
        return _wrap_tensor(_ops.mod(_to_tensor(self), _to_tensor(other)))

    def __rmod__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise remainder `other % self`.

        Args:
            other: Left-hand dividend operand.

        Returns:
            Tensor: Remainder tensor.
        """
        return _wrap_tensor(_ops.mod(_to_tensor(other), _to_tensor(self)))

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

    def __rpow__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise power `other ** self`.

        Args:
            other: Base operand.

        Returns:
            Tensor: Power tensor.
        """
        return _wrap_tensor(_ops.power(_to_tensor(other), _to_tensor(self)))

    def __and__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise bitwise/logical AND `self & other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Bitwise/logical AND tensor.
        """
        if self.dtype == DType.Bool:
            return _wrap_tensor(_ops.logical_and(_to_tensor(self), _to_tensor(other)))
        return _wrap_tensor(_msc_ops.bitwise_and(_to_tensor(self), _to_tensor(other)))

    def __rand__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise bitwise/logical AND `other & self`.

        Args:
            other: Left-hand operand.

        Returns:
            Tensor: Bitwise/logical AND tensor.
        """
        if self.dtype == DType.Bool:
            return _wrap_tensor(_ops.logical_and(_to_tensor(other), _to_tensor(self)))
        return _wrap_tensor(_msc_ops.bitwise_and(_to_tensor(other), _to_tensor(self)))

    def __or__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise bitwise/logical OR `self | other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Bitwise/logical OR tensor.
        """
        if self.dtype == DType.Bool:
            return _wrap_tensor(_ops.logical_or(_to_tensor(self), _to_tensor(other)))
        return _wrap_tensor(_msc_ops.bitwise_or(_to_tensor(self), _to_tensor(other)))

    def __ror__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise bitwise/logical OR `other | self`.

        Args:
            other: Left-hand operand.

        Returns:
            Tensor: Bitwise/logical OR tensor.
        """
        if self.dtype == DType.Bool:
            return _wrap_tensor(_ops.logical_or(_to_tensor(other), _to_tensor(self)))
        return _wrap_tensor(_msc_ops.bitwise_or(_to_tensor(other), _to_tensor(self)))

    def __xor__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise bitwise/logical XOR `self ^ other`.

        Args:
            other: Right-hand operand.

        Returns:
            Tensor: Bitwise/logical XOR tensor.
        """
        if self.dtype == DType.Bool:
            return _wrap_tensor(_ops.logical_xor(_to_tensor(self), _to_tensor(other)))
        return _wrap_tensor(_msc_ops.bitwise_xor(_to_tensor(self), _to_tensor(other)))

    def __rxor__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise bitwise/logical XOR `other ^ self`.

        Args:
            other: Left-hand operand.

        Returns:
            Tensor: Bitwise/logical XOR tensor.
        """
        if self.dtype == DType.Bool:
            return _wrap_tensor(_ops.logical_xor(_to_tensor(other), _to_tensor(self)))
        return _wrap_tensor(_msc_ops.bitwise_xor(_to_tensor(other), _to_tensor(self)))

    def __invert__(self) -> Tensor:
        """Compute element-wise bitwise/logical inversion `~self`.

        Returns:
            Tensor: Inverted tensor.
        """
        if self.dtype == DType.Bool:
            return _wrap_tensor(_ops.logical_not(_to_tensor(self)))
        return _wrap_tensor(_msc_ops.bitwise_not(_to_tensor(self)))

    def __lshift__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise bitwise left shift `self << other`.

        Args:
            other: Shift count operand.

        Returns:
            Tensor: Left-shifted tensor.
        """
        return _wrap_tensor(_msc_ops.left_shift(_to_tensor(self), _to_tensor(other)))

    def __rlshift__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise bitwise left shift `other << self`.

        Args:
            other: Left-hand value operand.

        Returns:
            Tensor: Left-shifted tensor.
        """
        return _wrap_tensor(_msc_ops.left_shift(_to_tensor(other), _to_tensor(self)))

    def __rshift__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute element-wise bitwise right shift `self >> other`.

        Args:
            other: Shift count operand.

        Returns:
            Tensor: Right-shifted tensor.
        """
        return _wrap_tensor(_msc_ops.right_shift(_to_tensor(self), _to_tensor(other)))

    def __rrshift__(self, other: Tensor | TensorOperand) -> Tensor:
        """Compute reflected element-wise bitwise right shift `other >> self`.

        Args:
            other: Left-hand value operand.

        Returns:
            Tensor: Right-shifted tensor.
        """
        return _wrap_tensor(_msc_ops.right_shift(_to_tensor(other), _to_tensor(self)))

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


def _apply_scatter_1d(
    base: list[NestedTensorData],
    indices: NestedTensorData,
    updates: NestedTensorData,
    mode: str,
) -> list[NestedTensorData]:
    """Apply 1-D or coordinate scatter updates onto `base` according to `mode`.

    Args:
        base: Mutable top-level Python list from variable tensor.
        indices: Index scalar or nested list of indices.
        updates: Update scalar or nested list of updates.
        mode: Scatter mode (`'update'`, `'add'`, or `'sub'`).

    Returns:
        list[NestedTensorData]: Updated list.
    """
    idx_list: list[int] = (
        [indices]
        if isinstance(indices, int)
        else [
            entry if isinstance(entry, int) else (entry[0] if isinstance(entry[0], int) else 0)
            for entry in indices
            if isinstance(entry, int) or (isinstance(entry, list) and bool(entry))
        ]
        if isinstance(indices, list)
        else []
    )
    upd_list: list[NestedTensorData] = updates if isinstance(updates, list) else [updates]
    for target_idx, upd_val in zip(idx_list, upd_list):
        cur_val = base[target_idx]
        if (
            mode == "add"
            and isinstance(cur_val, (int, float))
            and isinstance(upd_val, (int, float))
        ):
            base[target_idx] = cur_val + upd_val
        elif (
            mode == "sub"
            and isinstance(cur_val, (int, float))
            and isinstance(upd_val, (int, float))
        ):
            base[target_idx] = cur_val - upd_val
        else:
            base[target_idx] = upd_val
    return base


class _ValueDescriptor:
    """Descriptor exposing `Variable.value` as a callable `Variable` on both class and instances."""

    def __get__(
        self,
        instance: Variable | None,
        owner: type[Variable] | None = None,
    ) -> Variable:
        """Return a `Variable` sentinel on class access or `instance` on instance access.

        Args:
            instance: `Variable` instance, or `None` when accessed on `Variable`.
            owner: Owning `Variable` class.

        Returns:
            Variable: Bound `Variable` instance or class-level sentinel `Variable`.
        """
        del owner
        if instance is None:
            return Variable(0.0)
        return instance


class Variable(Tensor):
    """A mutable Tensor whose value can be updated via assign operations."""

    def __init__(
        self,
        initial_value: Tensor | TensorOperand,
        trainable: builtins.bool = True,
        name: str | None = None,
        dtype: DTypeLike | TensorOperand = None,
        synchronization: str = "AUTO",
        aggregation: str = "NONE",
    ) -> None:
        """Initialize a `Variable`.

        Args:
            initial_value: Initial value for the variable.
            trainable: Whether the variable is trainable.
            name: Optional name for the variable.
            dtype: Optional target data type.
            synchronization: Variable synchronization policy string.
            aggregation: Variable aggregation policy string.
        """
        super().__init__(initial_value, dtype=dtype)
        self._trainable = trainable
        self.name = name or "Variable:0"
        self._synchronization = synchronization
        self._aggregation = aggregation

    @property
    def trainable(self) -> builtins.bool:
        """Return whether this variable is trainable.

        Returns:
            bool: Trainable flag.
        """
        return self._trainable

    @trainable.setter
    def trainable(self, val: builtins.bool) -> None:
        """Set whether this variable is trainable.

        Args:
            val: New trainable boolean flag.
        """
        self._trainable = val

    @property
    def synchronization(self) -> str:
        """Return the variable synchronization policy.

        Returns:
            str: Synchronization policy string.
        """
        return self._synchronization

    @property
    def aggregation(self) -> str:
        """Return the variable aggregation policy.

        Returns:
            str: Aggregation policy string.
        """
        return self._aggregation

    value = _ValueDescriptor()

    def read_value(self) -> Tensor:
        """Return a `Tensor` snapshot of the current variable value.

        Returns:
            Tensor: Current variable value as a `Tensor`.
        """
        return Tensor(self._tensor)

    def initialized_value(self) -> Tensor:
        """Return the value of the initialized variable.

        Returns:
            Tensor: Current variable value as a `Tensor`.
        """
        return self.read_value()

    def load(
        self,
        value: Tensor | TensorOperand,
        session: TensorOperand = None,
    ) -> None:
        """Load a new value into this variable.

        Args:
            value: New value to load into the variable.
            session: Optional session parameter (ignored in eager execution).
        """
        del session
        self.assign(value)

    def count_up_to(self, limit: int) -> Tensor:
        """Increment this variable by `1` until it reaches `limit`.

        Args:
            limit: Exclusive upper bound for the counter variable.

        Returns:
            Tensor: Value of the variable prior to incrementing.

        Raises:
            IndexError: If the variable value has already reached `limit`.
        """
        cur = self.read_value()
        raw = cur.numpy()
        scalar = (
            int(float(str(raw.item())))
            if isinstance(raw, ArrayLikeProtocol)
            else int(float(str(raw)))
        )
        if scalar >= limit:
            raise IndexError(f"Variable count_up_to reached limit {limit}")
        self.assign_add(1)
        return cur

    def gather_nd(
        self,
        indices: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Gather slices from this variable according to `indices`.

        Args:
            indices: Index tensor or sequence.
            name: Optional operation name.

        Returns:
            Tensor: Gathered tensor slices.
        """
        del name
        return self.sparse_read(indices)

    def to_proto(self, export_scope: str | None = None) -> dict[str, str]:
        """Convert this `Variable` to a serialized dictionary representation.

        Args:
            export_scope: Optional scope prefix to strip from the variable name.

        Returns:
            dict[str, str]: Dictionary containing `variable_name` and `export_scope`.
        """
        return {"variable_name": self.name, "export_scope": export_scope or ""}

    @classmethod
    def from_proto(
        cls,
        variable_def: TensorOperand,
        import_scope: str | None = None,
    ) -> Variable:
        """Construct a `Variable` from a `variable_def` representation.

        Args:
            variable_def: Serialized variable definition or initial value.
            import_scope: Optional scope prefix to prepend to the variable name.

        Returns:
            Variable: Reconstructed `Variable` instance.
        """
        scope_prefix = f"{import_scope}/" if import_scope else ""
        init_val: TensorOperand = 0.0 if isinstance(variable_def, (str, dict)) else variable_def
        return cls(init_val, name=f"{scope_prefix}Variable:0")

    def assign(
        self,
        value: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
        read_value: builtins.bool = True,
    ) -> Variable | Tensor:
        """Assign a new value to this variable.

        Args:
            value: New value to assign.
            use_locking: If `True`, use locking during assignment.
            name: Optional name for the operation.
            read_value: If `True`, return updated `Variable`; else return a `Tensor` snapshot.

        Returns:
            Variable | Tensor: Updated variable (`self`) or `Tensor` snapshot.
        """
        del use_locking, name
        self._tensor = _to_tensor(value, dtype=self.dtype)
        return self if read_value else Tensor(self._tensor)

    def assign_add(
        self,
        delta: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
        read_value: builtins.bool = True,
    ) -> Variable | Tensor:
        """Add `delta` to this variable in-place.

        Args:
            delta: Tensor or scalar delta to add.
            use_locking: If `True`, use locking during update.
            name: Optional name for the operation.
            read_value: If `True`, return updated `Variable`; else return a `Tensor` snapshot.

        Returns:
            Variable | Tensor: Updated variable (`self`) or `Tensor` snapshot.
        """
        del use_locking, name
        self._tensor = _ops.add(_to_tensor(self), _to_tensor(delta))
        return self if read_value else Tensor(self._tensor)

    def assign_sub(
        self,
        delta: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
        read_value: builtins.bool = True,
    ) -> Variable | Tensor:
        """Subtract `delta` from this variable in-place.

        Args:
            delta: Tensor or scalar delta to subtract.
            use_locking: If `True`, use locking during update.
            name: Optional name for the operation.
            read_value: If `True`, return updated `Variable`; else return a `Tensor` snapshot.

        Returns:
            Variable | Tensor: Updated variable (`self`) or `Tensor` snapshot.
        """
        del use_locking, name
        self._tensor = _ops.subtract(_to_tensor(self), _to_tensor(delta))
        return self if read_value else Tensor(self._tensor)

    def _scatter_op(
        self,
        indices: Tensor | TensorOperand,
        updates: Tensor | TensorOperand,
        mode: str,
    ) -> Variable:
        """Apply in-place scatter operation (`mode`) to this variable.

        Args:
            indices: Indices tensor or sequence.
            updates: Updates tensor or sequence.
            mode: Scatter mode (`'update'`, `'add'`, or `'sub'`).

        Returns:
            Variable: Updated variable (`self`).
        """
        raw_base = self.numpy()
        py_base = raw_base.tolist() if isinstance(raw_base, ArrayLikeProtocol) else [raw_base]
        base_list: list[NestedTensorData] = py_base if isinstance(py_base, list) else [py_base]
        idx_t = Tensor(indices).numpy()
        py_idx = idx_t.tolist() if isinstance(idx_t, ArrayLikeProtocol) else idx_t
        upd_t = Tensor(updates).numpy()
        py_upd = upd_t.tolist() if isinstance(upd_t, ArrayLikeProtocol) else upd_t
        updated = _apply_scatter_1d(base_list, py_idx, py_upd, mode=mode)
        self._tensor = _to_tensor(updated, dtype=self.dtype)
        return self

    def scatter_nd_update(
        self,
        indices: Tensor | TensorOperand,
        updates: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Variable:
        """Apply sparse updates to individual values or slices in this variable.

        Args:
            indices: Index tensor or sequence.
            updates: Replacement values tensor or sequence.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del name
        return self._scatter_op(indices, updates, mode="update")

    def scatter_nd_add(
        self,
        indices: Tensor | TensorOperand,
        updates: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Variable:
        """Apply sparse addition to individual values or slices in this variable.

        Args:
            indices: Index tensor or sequence.
            updates: Delta values tensor or sequence to add.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del name
        return self._scatter_op(indices, updates, mode="add")

    def scatter_nd_sub(
        self,
        indices: Tensor | TensorOperand,
        updates: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Variable:
        """Apply sparse subtraction to individual values or slices in this variable.

        Args:
            indices: Index tensor or sequence.
            updates: Delta values tensor or sequence to subtract.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del name
        return self._scatter_op(indices, updates, mode="sub")

    def batch_scatter_update(
        self,
        sparse_delta: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
    ) -> Variable:
        """Assign `IndexedSlices` or tensor updates to this variable.

        Args:
            sparse_delta: Update tensor or sequence.
            use_locking: Whether to lock during assignment.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del use_locking, name
        self._tensor = _to_tensor(sparse_delta, dtype=self.dtype)
        return self

    def scatter_update(
        self,
        sparse_delta: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
    ) -> Variable:
        """Assign sparse delta updates to this variable.

        Args:
            sparse_delta: Sparse delta or tensor value to assign.
            use_locking: If `True`, use locking during the operation.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del use_locking, name
        self._tensor = _to_tensor(sparse_delta, dtype=self.dtype)
        return self

    def scatter_add(
        self,
        sparse_delta: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
    ) -> Variable:
        """Add sparse delta updates to this variable.

        Args:
            sparse_delta: Sparse delta or tensor value to add.
            use_locking: If `True`, use locking during the operation.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del use_locking, name
        self._tensor = _ops.add(_to_tensor(self), _to_tensor(sparse_delta))
        return self

    def scatter_sub(
        self,
        sparse_delta: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
    ) -> Variable:
        """Subtract sparse delta updates from this variable.

        Args:
            sparse_delta: Sparse delta or tensor value to subtract.
            use_locking: If `True`, use locking during the operation.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del use_locking, name
        self._tensor = _ops.subtract(_to_tensor(self), _to_tensor(sparse_delta))
        return self

    def scatter_mul(
        self,
        sparse_delta: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
    ) -> Variable:
        """Multiply this variable by sparse delta updates.

        Args:
            sparse_delta: Sparse delta or tensor value to multiply by.
            use_locking: If `True`, use locking during the operation.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del use_locking, name
        self._tensor = _ops.multiply(_to_tensor(self), _to_tensor(sparse_delta))
        return self

    def scatter_div(
        self,
        sparse_delta: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
    ) -> Variable:
        """Divide this variable by sparse delta updates.

        Args:
            sparse_delta: Sparse delta or tensor divisor.
            use_locking: If `True`, use locking during the operation.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del use_locking, name
        self._tensor = _ops.divide(_to_tensor(self), _to_tensor(sparse_delta))
        return self

    def scatter_max(
        self,
        sparse_delta: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
    ) -> Variable:
        """Update this variable with the element-wise maximum of `sparse_delta`.

        Args:
            sparse_delta: Sparse delta or tensor value.
            use_locking: If `True`, use locking during the operation.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del use_locking, name
        self._tensor = _ops.maximum(_to_tensor(self), _to_tensor(sparse_delta))
        return self

    def scatter_min(
        self,
        sparse_delta: Tensor | TensorOperand,
        use_locking: builtins.bool = False,
        name: str | None = None,
    ) -> Variable:
        """Update this variable with the element-wise minimum of `sparse_delta`.

        Args:
            sparse_delta: Sparse delta or tensor value.
            use_locking: If `True`, use locking during the operation.
            name: Optional operation name.

        Returns:
            Variable: Updated variable (`self`).
        """
        del use_locking, name
        self._tensor = _ops.minimum(_to_tensor(self), _to_tensor(sparse_delta))
        return self

    def sparse_read(
        self,
        indices: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Gather slices from this variable along axis 0 according to `indices`.

        Args:
            indices: Index integer or sequence of indices.
            name: Optional operation name.

        Returns:
            Tensor: Gathered tensor slices.
        """
        del name
        return _wrap_tensor(_ops.take(_to_tensor(self), _to_tensor(indices), axis=0))


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
    "_TensorRef",
    "_TracingContext",
    "_extract_constant_value",
    "_to_attribute_value",
    "_to_tensor",
    "_wrap",
    "_wrap_tensor",
    "to_dtype",
]
