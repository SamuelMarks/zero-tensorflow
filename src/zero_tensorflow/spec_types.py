"""Specification, graph, device, and container classes for `zero_tensorflow`."""

from __future__ import annotations

import builtins
import contextlib
import enum
import threading
from collections.abc import Callable, Iterator, Sequence
from types import TracebackType
from typing import Optional, TypeVar

from zero_tensorflow._typing import ArrayLikeProtocol, GradFnProtocol, TensorOperand
from zero_tensorflow.core_tensor import (
    _ACTIVE_MODULE_STACK,
    Tensor,
    Variable,
    _ModuleTrackerProtocol,
    _to_tensor,
    _wrap_tensor,
    to_dtype,
)
from zero_tensorflow.dtypes import DType
from zero_tensorflow.ragged import RaggedTensor
from zero_tensorflow.sparse import SparseTensor

from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.transforms.autodiff_rules.common import (
    UnconnectedGradients as _MscUnconnectedGradients,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from zero_keras import ops as _k_ops

__all__ = [
    "AggregationMethod",
    "CriticalSection",
    "DeviceSpec",
    "Graph",
    "IndexedSlices",
    "IndexedSlicesSpec",
    "Module",
    "Operation",
    "OptionalSpec",
    "RaggedTensorSpec",
    "RegisterGradient",
    "SparseTensorSpec",
    "TensorArray",
    "TensorArraySpec",
    "TensorShape",
    "TensorSpec",
    "TypeSpec",
    "UnconnectedGradients",
    "VariableAggregation",
    "VariableSynchronization",
    "type_spec_from_value",
]

_R = TypeVar("_R")
_GRADIENT_REGISTRY: dict[str, GradFnProtocol] = {}
_NAME_SCOPE_STACK: list[str] = []


def _normalize_dims(
    dims: TensorShape | Sequence[int | None] | int | TensorOperand,
) -> tuple[tuple[int | None, ...], bool]:
    """Normalize a shape specification into `(dims_tuple, is_unknown_rank)`.

    Args:
        dims: Input `TensorShape`, sequence of dimensions, scalar dimension, or `None`.

    Returns:
        tuple[tuple[int | None, ...], bool]: Normalized dimensions and unknown-rank flag.
    """
    if isinstance(dims, TensorShape):
        return (builtins.tuple(dims), dims._unknown_rank)
    if dims is None:
        return ((), True)
    if isinstance(dims, Tensor):
        dims = dims.numpy()
    if isinstance(dims, ArrayLikeProtocol):
        dims = dims.tolist()
    if isinstance(dims, int) and not isinstance(dims, bool):
        return ((dims,), False)
    if isinstance(dims, Sequence) and not isinstance(dims, (str, bytes)):
        parsed: list[int | None] = [
            int(d) if isinstance(d, int) and not isinstance(d, bool) else None for d in dims
        ]
        return (builtins.tuple(parsed), False)
    return ((), False)


class TensorShape(builtins.tuple[Optional[int], ...]):
    """Represents the shape of a `Tensor`."""

    _unknown_rank: bool

    def __new__(
        cls,
        dims: TensorShape | Sequence[int | None] | int | TensorOperand = (),
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> TensorShape:
        """Create a new `TensorShape` instance.

        Args:
            dims: A sequence of dimensions, an `int`, a `TensorShape`, or `None` for unknown rank.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorShape: Constructed `TensorShape`.
        """
        del args, kwargs
        norm_dims, unknown = _normalize_dims(dims)
        instance = super().__new__(cls, norm_dims)
        instance._unknown_rank = unknown
        return instance

    @property
    def dims(self) -> list[int | None] | None:
        """Return list of dimensions, or `None` if rank is unknown.

        Returns:
            list[int | None] | None: Dimension list or `None`.
        """
        return None if self._unknown_rank else list(self)

    @property
    def ndims(self) -> int | None:
        """Return rank of this shape, or `None` if unknown.

        Returns:
            int | None: Number of dimensions or `None`.
        """
        return None if self._unknown_rank else len(self)

    @property
    def rank(self) -> int | None:
        """Return rank of this shape, or `None` if unknown.

        Returns:
            int | None: Number of dimensions or `None`.
        """
        return self.ndims

    def as_list(self) -> list[int | None]:
        """Return a list of integers or `None` for each dimension.

        Returns:
            list[int | None]: Dimension sizes.

        Raises:
            ValueError: If `self` has unknown rank (`dims is None`).
        """
        if self._unknown_rank:
            raise ValueError("as_list() is not defined on an unknown TensorShape.")
        return list(self)

    def is_compatible_with(
        self,
        other: TensorShape | Sequence[int | None] | int | TensorOperand,
    ) -> bool:
        """Return `True` if `self` is compatible with `other`.

        Args:
            other: Another `TensorShape` or shape-like sequence.

        Returns:
            bool: `True` if shapes have the same rank and compatible dimensions.
        """
        other_shape = other if isinstance(other, TensorShape) else TensorShape(other)
        if self._unknown_rank or other_shape._unknown_rank:
            return True
        if len(self) != len(other_shape):
            return False
        return all(d1 is None or d2 is None or d1 == d2 for d1, d2 in zip(self, other_shape))

    def merge_with(
        self,
        other: TensorShape | Sequence[int | None] | int | TensorOperand,
    ) -> TensorShape:
        """Return a `TensorShape` combining the information in `self` and `other`.

        Args:
            other: Another `TensorShape` or shape-like sequence.

        Returns:
            TensorShape: Merged `TensorShape`.

        Raises:
            ValueError: If `self` and `other` are not compatible.
        """
        other_shape = other if isinstance(other, TensorShape) else TensorShape(other)
        if not self.is_compatible_with(other_shape):
            raise ValueError(f"Shapes {self} and {other_shape} are not compatible.")
        if self._unknown_rank:
            return other_shape
        if other_shape._unknown_rank:
            return self
        return TensorShape([d1 if d1 is not None else d2 for d1, d2 in zip(self, other_shape)])

    def concatenate(
        self,
        other: TensorShape | Sequence[int | None] | int | TensorOperand,
    ) -> TensorShape:
        """Return the concatenation of the dimension in `self` and `other`.

        Args:
            other: Another `TensorShape` or shape-like sequence.

        Returns:
            TensorShape: Concatenated `TensorShape`.
        """
        other_shape = other if isinstance(other, TensorShape) else TensorShape(other)
        if self._unknown_rank or other_shape._unknown_rank:
            return TensorShape(None)
        return TensorShape([*self, *other_shape])

    def is_fully_defined(self) -> bool:
        """Return `True` iff `self` is known rank and every dimension is specified.

        Returns:
            bool: `True` if rank and all dimensions are non-`None` integers.
        """
        return not self._unknown_rank and all(isinstance(d, int) for d in self)

    def assert_is_fully_defined(self) -> None:
        """Raise `ValueError` if `self` is not fully defined.

        Raises:
            ValueError: If `self` has unknown rank or any `None` dimension.
        """
        if not self.is_fully_defined():
            raise ValueError(f"Shape {self} is not fully defined")

    def assert_has_rank(self, rank: int) -> None:
        """Raise `ValueError` if `self` is not compatible with the given `rank`.

        Args:
            rank: Expected rank integer.

        Raises:
            ValueError: If `self` has a known rank not equal to `rank`.
        """
        if not self._unknown_rank and len(self) != rank:
            raise ValueError(f"Shape {self} must have rank {rank}")

    def assert_is_compatible_with(
        self,
        other: TensorShape | Sequence[int | None] | int | TensorOperand,
    ) -> None:
        """Raise `ValueError` if `self` is not compatible with `other`.

        Args:
            other: Another `TensorShape` or shape-like sequence.

        Raises:
            ValueError: If `self` and `other` are incompatible.
        """
        if not self.is_compatible_with(other):
            raise ValueError(f"Shapes {self} and {other!r} are incompatible")

    def with_rank(self, rank: int) -> TensorShape:
        """Return a shape based on `self` with the given `rank`.

        Args:
            rank: Required rank integer.

        Returns:
            TensorShape: Merged `TensorShape` of rank `rank`.
        """
        return self.merge_with(TensorShape([None] * rank))

    def with_rank_at_least(self, rank: int) -> TensorShape:
        """Return `self` if its rank is unknown or at least `rank`.

        Args:
            rank: Minimum required rank.

        Returns:
            TensorShape: `self`.

        Raises:
            ValueError: If `self` has known rank `< rank`.
        """
        if not self._unknown_rank and len(self) < rank:
            raise ValueError(f"Shape {self} must have rank at least {rank}")
        return self

    def with_rank_at_most(self, rank: int) -> TensorShape:
        """Return `self` if its rank is unknown or at most `rank`.

        Args:
            rank: Maximum allowed rank.

        Returns:
            TensorShape: `self`.

        Raises:
            ValueError: If `self` has known rank `> rank`.
        """
        if not self._unknown_rank and len(self) > rank:
            raise ValueError(f"Shape {self} must have rank at most {rank}")
        return self

    def num_elements(self) -> int | None:
        """Return the total number of elements, or `None` if shape is not fully defined.

        Returns:
            int | None: Product of dimensions if fully defined, else `None`.
        """
        if not self.is_fully_defined():
            return None
        prod = 1
        for d in self:
            assert isinstance(d, int)
            prod *= d
        return prod


class TypeSpec:
    """Specifies a TensorFlow value type."""

    @property
    def value_type(self) -> type:
        """Return the Python type for values that are compatible with this `TypeSpec`.

        Returns:
            type: Compatible Python value class.
        """
        return Tensor

    def is_compatible_with(self, spec_or_value: TypeSpec | Tensor | TensorOperand) -> bool:
        """Return `True` if `spec_or_value` is compatible with this `TypeSpec`.

        Args:
            spec_or_value: Candidate `TypeSpec` or value.

        Returns:
            bool: `True` if compatible.
        """
        return isinstance(spec_or_value, (type(self), self.value_type))

    def most_specific_compatible_type(self, other: TypeSpec) -> TypeSpec:
        """Return the most specific `TypeSpec` compatible with both `self` and `other`.

        Args:
            other: Another `TypeSpec`.

        Returns:
            TypeSpec: Compatible `TypeSpec`.

        Raises:
            ValueError: If `other` is not compatible with `self`.
        """
        if not isinstance(other, type(self)):
            raise ValueError(f"Types {self} and {other} are not compatible.")
        return self


class TensorSpec(TypeSpec):
    """Describes the shape and dtype of a dense `tf.Tensor`."""

    def __init__(
        self,
        shape: TensorShape | Sequence[int | None] | int | TensorOperand = (),
        dtype: DType | str | TensorOperand = DType.Float32,
        name: str | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `TensorSpec`.

        Args:
            shape: Shape of the tensor (`TensorShape`, tuple, list, or `None`).
            dtype: Data type of the tensor (`DType` or string alias).
            name: Optional name for the tensor specification.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self.shape: TensorShape = shape if isinstance(shape, TensorShape) else TensorShape(shape)
        self.dtype: DType = to_dtype(dtype)
        self.name: str | None = name

    @property
    def value_type(self) -> type[Tensor]:
        """Return the Python type represented by `TensorSpec`.

        Returns:
            type[Tensor]: `Tensor` class.
        """
        return Tensor

    def is_compatible_with(
        self,
        spec_or_tensor: TypeSpec | Tensor | Variable | TensorOperand,
    ) -> bool:
        """Return `True` if `spec_or_tensor` is compatible with this `TensorSpec`.

        Args:
            spec_or_tensor: Candidate `TensorSpec`, `Tensor`, or tensor-like value.

        Returns:
            bool: `True` if both `dtype` and `shape` are compatible.
        """
        if isinstance(spec_or_tensor, TensorSpec):
            return self.dtype == spec_or_tensor.dtype and self.shape.is_compatible_with(
                spec_or_tensor.shape
            )
        if isinstance(spec_or_tensor, (Tensor, Variable)):
            t_dt = to_dtype(spec_or_tensor.dtype)
            return t_dt == self.dtype and self.shape.is_compatible_with(spec_or_tensor.shape)
        return False

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor | Variable | TensorOperand,
        name: str | None = None,
    ) -> TensorSpec:
        """Return a `TensorSpec` that describes `tensor`.

        Args:
            tensor: Input `Tensor` or `Variable`.
            name: Optional specification name.

        Returns:
            TensorSpec: Specification matching `tensor`'s shape and dtype.
        """
        t = tensor if isinstance(tensor, (Tensor, Variable)) else Tensor(tensor)
        return cls(shape=t.shape, dtype=t.dtype or DType.Float32, name=name)

    from_value = from_tensor


class RaggedTensorSpec(TypeSpec):
    """Type specification for a `tf.RaggedTensor`."""

    def __init__(
        self,
        shape: TensorShape | Sequence[int | None] | TensorOperand = None,
        dtype: DType | str | TensorOperand = DType.Float32,
        ragged_rank: int | None = None,
        row_splits_dtype: DType | str | TensorOperand = DType.Int64,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `RaggedTensorSpec`.

        Args:
            shape: Shape of the ragged tensor.
            dtype: Data type of flat values.
            ragged_rank: Number of ragged dimensions.
            row_splits_dtype: Data type of `row_splits` (`int32` or `int64`).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self.shape = shape if isinstance(shape, TensorShape) else TensorShape(shape)
        self.dtype = to_dtype(dtype) or DType.Float32
        self.ragged_rank = ragged_rank if ragged_rank is not None else 1
        self.row_splits_dtype = to_dtype(row_splits_dtype) or DType.Int64

    @property
    def value_type(self) -> type[RaggedTensor]:
        """Return `RaggedTensor` as the value type.

        Returns:
            type[RaggedTensor]: `RaggedTensor` class.
        """
        return RaggedTensor

    @classmethod
    def from_value(cls, value: RaggedTensor) -> RaggedTensorSpec:
        """Construct a `RaggedTensorSpec` from a `RaggedTensor`.

        Args:
            value: Input `RaggedTensor`.

        Returns:
            RaggedTensorSpec: Specification describing `value`.
        """
        return cls(
            shape=value.shape,
            dtype=value.dtype,
            ragged_rank=value.ragged_rank,
        )


class SparseTensorSpec(TypeSpec):
    """Type specification for a `tf.SparseTensor`."""

    def __init__(
        self,
        shape: TensorShape | Sequence[int | None] | TensorOperand = None,
        dtype: DType | str | TensorOperand = DType.Float32,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `SparseTensorSpec`.

        Args:
            shape: Dense shape of the sparse tensor.
            dtype: Data type of non-zero values.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self.shape = shape if isinstance(shape, TensorShape) else TensorShape(shape)
        self.dtype = to_dtype(dtype) or DType.Float32

    @property
    def value_type(self) -> type[SparseTensor]:
        """Return `SparseTensor` as the value type.

        Returns:
            type[SparseTensor]: `SparseTensor` class.
        """
        return SparseTensor

    @classmethod
    def from_value(cls, value: SparseTensor) -> SparseTensorSpec:
        """Construct a `SparseTensorSpec` from a `SparseTensor`.

        Args:
            value: Input `SparseTensor`.

        Returns:
            SparseTensorSpec: Specification describing `value`.
        """
        return cls(shape=value.shape, dtype=_to_tensor(value.values).dtype)


class IndexedSlices:
    """Sparse representation of a set of tensor slices at given indices."""

    def __init__(
        self,
        values: Tensor | TensorOperand = None,
        indices: Tensor | TensorOperand = None,
        dense_shape: Tensor | Sequence[int] | TensorOperand = None,
        name: str | None = None,
    ) -> None:
        """Initialize `IndexedSlices`.

        Args:
            values: `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
            indices: 1-D integer `Tensor` with shape `[D0]`.
            dense_shape: Optional 1-D integer `Tensor` or tuple containing the dense shape.
            name: Optional name for the `IndexedSlices`.
        """
        self.values: Tensor = (
            values if isinstance(values, Tensor) else Tensor(values if values is not None else [])
        )
        self.indices: Tensor = (
            indices
            if isinstance(indices, Tensor)
            else Tensor(indices if indices is not None else [], dtype=DType.Int32)
        )
        self.dense_shape: Tensor | None = (
            (
                dense_shape
                if isinstance(dense_shape, Tensor)
                else Tensor(dense_shape, dtype=DType.Int32)
            )
            if dense_shape is not None
            else None
        )
        self.name = name

    @property
    def shape(self) -> builtins.tuple[int, ...]:
        """Return the dense shape if known, else `self.values.shape`.

        Returns:
            tuple[int, ...]: Shape tuple.
        """
        if self.dense_shape is not None:
            raw = self.dense_shape.numpy()
            raw_list = raw.tolist() if isinstance(raw, ArrayLikeProtocol) else raw
            if isinstance(raw_list, list):
                return builtins.tuple(int(d) for d in raw_list if isinstance(d, (int, float)))
        return self.values.shape

    @property
    def dtype(self) -> DType | None:
        """Return the `DType` of `self.values`.

        Returns:
            DType | None: Value data type.
        """
        return self.values.dtype

    @property
    def device(self) -> str:
        """Return the device of `self.values`.

        Returns:
            str: Device string.
        """
        return self.values.device


class IndexedSlicesSpec(TypeSpec):
    """Type specification for `tf.IndexedSlices`."""

    def __init__(
        self,
        shape: TensorShape | Sequence[int | None] | TensorOperand = None,
        dtype: DType | str | TensorOperand = DType.Float32,
        indices_dtype: DType | str | TensorOperand = DType.Int64,
        dense_shape_dtype: DType | str | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `IndexedSlicesSpec`.

        Args:
            shape: Dense shape of the `IndexedSlices`.
            dtype: Data type of slice values.
            indices_dtype: Data type of slice indices.
            dense_shape_dtype: Optional data type of `dense_shape`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self.shape = shape if isinstance(shape, TensorShape) else TensorShape(shape)
        self.dtype = to_dtype(dtype) or DType.Float32
        self.indices_dtype = to_dtype(indices_dtype) or DType.Int64
        self.dense_shape_dtype = (
            to_dtype(dense_shape_dtype) if dense_shape_dtype is not None else None
        )

    @property
    def value_type(self) -> type[IndexedSlices]:
        """Return `IndexedSlices` as the value type.

        Returns:
            type[IndexedSlices]: `IndexedSlices` class.
        """
        return IndexedSlices


class OptionalSpec(TypeSpec):
    """Type specification for `tf.experimental.Optional` / `tf.data.experimental.Optional`."""

    def __init__(
        self,
        element_spec: TypeSpec | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `OptionalSpec`.

        Args:
            element_spec: Type specification of the wrapped optional element.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self.element_spec = element_spec

    @classmethod
    def from_value(cls, value: TensorOperand) -> OptionalSpec:
        """Create an `OptionalSpec` from an optional value container.

        Args:
            value: Input optional value.

        Returns:
            OptionalSpec: Constructed `OptionalSpec`.
        """
        return cls(element_spec=type_spec_from_value(value))


class TensorArray:
    """Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays."""

    def __init__(
        self,
        dtype: DType | str | TensorOperand = DType.Float32,
        size: int = 0,
        dynamic_size: bool = False,
        clear_after_read: bool = True,
        tensor_array_name: str | None = None,
        handle: TensorOperand = None,
        flow: TensorOperand = None,
        infer_shape: bool = True,
        element_shape: TensorShape | Sequence[int | None] | None = None,
        colocate_with_first_write_call: bool = True,
        name: str | None = None,
    ) -> None:
        """Initialize `TensorArray`.

        Args:
            dtype: Data type of elements in the array.
            size: Initial array size.
            dynamic_size: Whether writes past `size` dynamically grow the array.
            clear_after_read: Whether reading an element clears its slot.
            tensor_array_name: Optional array name.
            handle: Optional handle tensor.
            flow: Optional control-flow float scalar.
            infer_shape: Whether to infer uniform element shape from writes.
            element_shape: Optional expected shape of each element.
            colocate_with_first_write_call: Whether to colocate on first write.
            name: Optional operation name.
        """
        self.dtype: DType = to_dtype(dtype)
        self._size: int = max(0, size)
        self.dynamic_size = dynamic_size
        self.clear_after_read = clear_after_read
        self.tensor_array_name = tensor_array_name
        self.handle = handle
        self.flow = flow if isinstance(flow, Tensor) else Tensor(0.0, dtype=DType.Float32)
        self.infer_shape = infer_shape
        self.element_shape: TensorShape = (
            element_shape if isinstance(element_shape, TensorShape) else TensorShape(element_shape)
        )
        self.colocate_with_first_write_call = colocate_with_first_write_call
        self.name = name
        self._closed: bool = False
        self._buffer: list[Tensor | None] = [None] * self._size

    def size(self, name: str | None = None) -> Tensor:
        """Return the current number of slots in the `TensorArray`.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Scalar `int32` tensor containing the array size.
        """
        del name
        return Tensor(len(self._buffer), dtype=DType.Int32)

    def write(
        self,
        index: int,
        value: Tensor | TensorOperand,
        name: str | None = None,
    ) -> TensorArray:
        """Write `value` into `index` of the `TensorArray`.

        Args:
            index: Non-negative integer slot index.
            value: Value to write as a `Tensor`.
            name: Optional operation name.

        Returns:
            TensorArray: `self` with updated slot.

        Raises:
            IndexError: If `index` is out of bounds and `dynamic_size` is `False`.
        """
        del name
        if index < 0 or (index >= len(self._buffer) and not self.dynamic_size):
            raise IndexError(
                f"TensorArray index {index} out of bounds for size {len(self._buffer)}."
            )
        while index >= len(self._buffer):
            self._buffer.append(None)
        t_val = _wrap_tensor(_to_tensor(value, dtype=self.dtype))
        if self.infer_shape and self.element_shape.rank is None:
            self.element_shape = TensorShape(t_val.shape)
        self._buffer[index] = t_val
        return self

    def read(self, index: int, name: str | None = None) -> Tensor:
        """Read the value at `index` in the `TensorArray`.

        Args:
            index: Slot index to read.
            name: Optional operation name.

        Returns:
            Tensor: Tensor stored at `index` (or scalar zero if unwritten).

        Raises:
            IndexError: If `index` is out of bounds.
        """
        del name
        if index < 0 or index >= len(self._buffer):
            raise IndexError(
                f"TensorArray index {index} out of bounds for size {len(self._buffer)}."
            )
        item = self._buffer[index]
        res = item if item is not None else Tensor(0, dtype=self.dtype)
        if self.clear_after_read:
            self._buffer[index] = None
        return res

    def stack(self, name: str | None = None) -> Tensor:
        """Return the values in the `TensorArray` as a stacked `Tensor`.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Stacked tensor across all slots.
        """
        del name
        if not self._buffer:
            return Tensor([], dtype=self.dtype)
        elem_dims = [d if isinstance(d, int) else 1 for d in self.element_shape]
        default_t = (
            _k_ops.zeros(builtins.tuple(elem_dims), dtype=self.dtype.value)
            if elem_dims
            else _to_tensor(Tensor(0, dtype=self.dtype))
        )
        tensors = [_to_tensor(item) if item is not None else default_t for item in self._buffer]
        return _wrap_tensor(_k_ops.stack(tensors, axis=0))

    def unstack(
        self,
        value: Tensor | Sequence[TensorOperand],
        name: str | None = None,
    ) -> TensorArray:
        """Populate the `TensorArray` by unstacking `value` along dimension 0.

        Args:
            value: N-D `Tensor` or sequence to unstack.
            name: Optional operation name.

        Returns:
            TensorArray: `self` populated with unstacked slices.
        """
        del name
        slices = _k_ops.unstack(_to_tensor(value, dtype=self.dtype), axis=0)
        self._buffer = [_wrap_tensor(s) for s in slices]
        self._size = len(self._buffer)
        return self

    def concat(self, name: str | None = None) -> Tensor:
        """Concatenate all elements in the `TensorArray` along dimension 0.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Concatenated tensor.
        """
        del name
        valid = [_to_tensor(item) for item in self._buffer if item is not None]
        if not valid:
            return Tensor([], dtype=self.dtype)
        return _wrap_tensor(_k_ops.concatenate(valid, axis=0))

    def gather(
        self,
        indices: Sequence[int] | Tensor,
        name: str | None = None,
    ) -> Tensor:
        """Return selected elements from the `TensorArray` stacked into a `Tensor`.

        Args:
            indices: 1-D sequence or `Tensor` of slot indices.
            name: Optional operation name.

        Returns:
            Tensor: Gathered and stacked `Tensor`.
        """
        del name
        raw_idx = indices.numpy() if isinstance(indices, Tensor) else indices
        unwrapped_idx = raw_idx.tolist() if isinstance(raw_idx, ArrayLikeProtocol) else raw_idx
        idx_list = (
            [int(i) for i in unwrapped_idx if isinstance(i, (int, float))]
            if isinstance(unwrapped_idx, Sequence) and not isinstance(unwrapped_idx, (str, bytes))
            else []
        )
        picked = [
            _to_tensor(
                self._buffer[i]
                if 0 <= i < len(self._buffer) and self._buffer[i] is not None
                else Tensor(0, dtype=self.dtype)
            )
            for i in idx_list
        ]
        if not picked:
            return Tensor([], dtype=self.dtype)
        return _wrap_tensor(_k_ops.stack(picked, axis=0))

    def scatter(
        self,
        indices: Sequence[int] | Tensor,
        value: Tensor | Sequence[TensorOperand],
        name: str | None = None,
    ) -> TensorArray:
        """Scatter slices of `value` at `indices` into the `TensorArray`.

        Args:
            indices: 1-D sequence or `Tensor` of target slot indices.
            value: Tensor or sequence whose leading dimension matches `len(indices)`.
            name: Optional operation name.

        Returns:
            TensorArray: `self` after scattering values.
        """
        del name
        raw_idx = indices.numpy() if isinstance(indices, Tensor) else indices
        unwrapped_idx = raw_idx.tolist() if isinstance(raw_idx, ArrayLikeProtocol) else raw_idx
        idx_list = (
            [int(i) for i in unwrapped_idx if isinstance(i, (int, float))]
            if isinstance(unwrapped_idx, Sequence) and not isinstance(unwrapped_idx, (str, bytes))
            else []
        )
        slices = _k_ops.unstack(_to_tensor(value, dtype=self.dtype), axis=0)
        for idx, sl in zip(idx_list, slices):
            self.write(idx, _wrap_tensor(sl))
        return self

    def close(self, name: str | None = None) -> None:
        """Close the `TensorArray` and release buffered tensors.

        Args:
            name: Optional operation name.
        """
        del name
        self._closed = True
        self._buffer.clear()

    def identity(self) -> TensorArray:
        """Return a shallow reference to `self`.

        Returns:
            TensorArray: `self`.
        """
        return self


class TensorArraySpec(TypeSpec):
    """Type specification for a `tf.TensorArray`."""

    def __init__(
        self,
        element_shape: TensorShape | Sequence[int | None] | TensorOperand = None,
        dtype: DType | str | TensorOperand = DType.Float32,
        dynamic_size: bool = False,
        infer_shape: bool = True,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `TensorArraySpec`.

        Args:
            element_shape: Shape of each element in the `TensorArray`.
            dtype: Data type of the elements.
            dynamic_size: Whether the `TensorArray` can dynamically grow.
            infer_shape: Whether element shapes are inferred.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self.element_shape = (
            element_shape if isinstance(element_shape, TensorShape) else TensorShape(element_shape)
        )
        self.dtype = to_dtype(dtype) or DType.Float32
        self.dynamic_size = dynamic_size
        self.infer_shape = infer_shape

    @property
    def value_type(self) -> type[TensorArray]:
        """Return `TensorArray` as the value type.

        Returns:
            type[TensorArray]: `TensorArray` class.
        """
        return TensorArray


def type_spec_from_value(
    value: Tensor
    | Variable
    | RaggedTensor
    | SparseTensor
    | IndexedSlices
    | TensorArray
    | TensorOperand = None,
) -> TypeSpec:
    """Return a `TypeSpec` that represents the given `value`.

    Args:
        value: A value that can be accepted or returned by TensorFlow APIs.

    Returns:
        TypeSpec: Concrete `TypeSpec` subclass describing `value`.
    """
    if isinstance(value, RaggedTensor):
        return RaggedTensorSpec.from_value(value)
    if isinstance(value, SparseTensor):
        return SparseTensorSpec.from_value(value)
    if isinstance(value, IndexedSlices):
        return IndexedSlicesSpec(shape=value.shape, dtype=value.dtype or DType.Float32)
    if isinstance(value, TensorArray):
        return TensorArraySpec(
            element_shape=value.element_shape,
            dtype=value.dtype,
            dynamic_size=value.dynamic_size,
            infer_shape=value.infer_shape,
        )
    return TensorSpec.from_tensor(value if value is not None else 0.0)


class Operation:
    """Represents a graph node that performs computation on tensors."""

    def __init__(
        self,
        node_def: LogicalNode | str | TensorOperand = None,
        op_type: str = "NoOp",
        inputs: Sequence[Tensor] | None = None,
        outputs: Sequence[Tensor] | None = None,
        device: str = "/device:CPU:0",
        name: str | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Operation`.

        Args:
            node_def: `LogicalNode` or operation name string.
            op_type: Operation type name (e.g., `'MatMul'`, `'Add'`, `'NoOp'`).
            inputs: Sequence of input `Tensor` objects.
            outputs: Sequence of output `Tensor` objects.
            device: Device specification string.
            name: Optional operation name.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        if isinstance(node_def, LogicalNode):
            self._node = node_def
        else:
            node_id = (
                node_def if isinstance(node_def, str) and node_def else (name if name else op_type)
            )
            self._node = LogicalNode(id=node_id, op_type=op_type)
        self._inputs: list[Tensor] = list(inputs) if inputs is not None else []
        self._outputs: list[Tensor] = list(outputs) if outputs is not None else []
        self._device = device

    @property
    def name(self) -> str:
        """Return the full name of this operation.

        Returns:
            str: Operation name.
        """
        return self._node.id

    @property
    def type(self) -> str:
        """Return the operation type string (e.g. `'MatMul'`).

        Returns:
            str: Operation type.
        """
        return self._node.op_type

    @property
    def inputs(self) -> list[Tensor]:
        """Return the list of `Tensor` objects representing the inputs of this op.

        Returns:
            list[Tensor]: Input tensors.
        """
        return list(self._inputs)

    @property
    def outputs(self) -> list[Tensor]:
        """Return the list of `Tensor` objects representing the outputs of this op.

        Returns:
            list[Tensor]: Output tensors.
        """
        return list(self._outputs)

    @property
    def device(self) -> str:
        """Return the device string of this operation.

        Returns:
            str: Device string.
        """
        return self._device

    @property
    def node_def(self) -> LogicalNode:
        """Return the underlying `LogicalNode` IR representation.

        Returns:
            LogicalNode: IR node definition.
        """
        return self._node


class _GraphDefaultContext:
    """Context manager installing a `Graph` as the active tracing graph."""

    def __init__(self, graph: Graph) -> None:
        """Initialize `_GraphDefaultContext`.

        Args:
            graph: Target `Graph` instance.
        """
        self._graph = graph
        self._prev_graph: LogicalGraph | None = None

    def __enter__(self) -> Graph:
        """Activate `self._graph` on the global tracing state.

        Returns:
            Graph: Activated `Graph`.
        """
        self._prev_graph = _tracer.active_graph
        _tracer.active_graph = self._graph._logical_graph
        return self._graph

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restore the previous active graph.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception instance if raised.
            exc_tb: Traceback if raised.
        """
        del exc_type, exc_val, exc_tb
        _tracer.active_graph = self._prev_graph


class Graph:
    """A TensorFlow computation, represented as a dataflow `LogicalGraph`."""

    def __init__(self, name: str = "Graph", *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `Graph`.

        Args:
            name: Optional graph identifier.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self._logical_graph = LogicalGraph(name=name)

    def as_default(self) -> _GraphDefaultContext:
        """Return a context manager that makes this `Graph` the default graph.

        Returns:
            _GraphDefaultContext: Graph activation context manager.
        """
        return _GraphDefaultContext(self)

    def as_graph_def(self) -> LogicalGraph:
        """Return the underlying `LogicalGraph` representation of this graph.

        Returns:
            LogicalGraph: IR graph container.
        """
        return self._logical_graph

    def get_operations(self) -> list[Operation]:
        """Return the list of operations in the graph.

        Returns:
            list[Operation]: Operations corresponding to nodes in the graph.
        """
        return [
            Operation(node, op_type=node.op_type) for node in self._logical_graph.nodes.values()
        ]

    def get_operation_by_name(self, name: str) -> Operation:
        """Return the `Operation` with the given `name`.

        Args:
            name: Operation name to look up.

        Returns:
            Operation: Matching `Operation`.

        Raises:
            KeyError: If `name` is not in the graph.
        """
        if name not in self._logical_graph.nodes:
            raise KeyError(f"The name '{name}' refers to an Operation not in the graph.")
        node = self._logical_graph.nodes[name]
        return Operation(node, op_type=node.op_type)

    def get_tensor_by_name(self, name: str) -> Tensor:
        """Return the `Tensor` with the given `name` (e.g. `'op_name:0'`).

        Args:
            name: Tensor name to look up.

        Returns:
            Tensor: Symbolic `Tensor` backed by the named graph node.

        Raises:
            KeyError: If the node is not found in the graph.
        """
        node_name = name.split(":")[0]
        if node_name not in self._logical_graph.nodes:
            raise KeyError(f"The name '{name}' refers to a Tensor which does not exist.")
        return Tensor(_traced_node=self._logical_graph.nodes[node_name])

    @contextlib.contextmanager
    def name_scope(self, name: str | None) -> Iterator[str]:
        """Return a context manager that creates hierarchical names for operations.

        Args:
            name: Scope prefix string.

        Yields:
            str: Active hierarchical scope path.
        """
        clean = name.strip("/") if isinstance(name, str) else ""
        if clean:
            _NAME_SCOPE_STACK.append(clean)
        try:
            yield "/".join(_NAME_SCOPE_STACK) + ("/" if _NAME_SCOPE_STACK else "")
        finally:
            if clean:
                _NAME_SCOPE_STACK.pop()

    @contextlib.contextmanager
    def control_dependencies(
        self,
        control_inputs: Sequence[Operation | Tensor] | None,
    ) -> Iterator[None]:
        """Return a context manager that specifies control dependencies.

        Args:
            control_inputs: List of `Operation` or `Tensor` objects.

        Yields:
            None: Control dependency scope.
        """
        del control_inputs
        yield

    @contextlib.contextmanager
    def device(self, device_name_or_function: str | None) -> Iterator[None]:
        """Return a context manager that specifies the default device.

        Args:
            device_name_or_function: Device specification string.

        Yields:
            None: Device scope.
        """
        del device_name_or_function
        yield


class DeviceSpec:
    """Represents a (possibly partial) specification for a TensorFlow device."""

    def __init__(
        self,
        job: str | None = None,
        replica: int | None = None,
        task: int | None = None,
        device_type: str | None = None,
        device_index: int | None = None,
    ) -> None:
        """Initialize `DeviceSpec`.

        Args:
            job: Optional job name string.
            replica: Optional replica index integer.
            task: Optional task index integer.
            device_type: Optional device type string (e.g., `'CPU'`, `'GPU'`).
            device_index: Optional device index integer.
        """
        self.job = job
        self.replica = replica
        self.task = task
        self.device_type = device_type.upper() if isinstance(device_type, str) else None
        self.device_index = device_index

    @classmethod
    def from_string(cls, spec: str) -> DeviceSpec:
        """Construct a `DeviceSpec` from a string representation.

        Args:
            spec: Device string of the form `'/job:.../replica:.../task:.../device:CPU:0'`.

        Returns:
            DeviceSpec: Parsed `DeviceSpec`.
        """
        return cls().parse_from_string(spec)

    def parse_from_string(self, spec: str) -> DeviceSpec:
        """Parse a `DeviceSpec` name into its components.

        Args:
            spec: Device specification string.

        Returns:
            DeviceSpec: Updated `DeviceSpec` instance (`self`).
        """
        self.job = None
        self.replica = None
        self.task = None
        self.device_type = None
        self.device_index = None
        parts = [p for p in spec.split("/") if p]
        for part in parts:
            tokens = part.split(":")
            key = tokens[0].lower()
            if key == "job" and len(tokens) >= 2:
                self.job = tokens[1]
            elif key == "replica" and len(tokens) >= 2 and tokens[1].isdigit():
                self.replica = int(tokens[1])
            elif key == "task" and len(tokens) >= 2 and tokens[1].isdigit():
                self.task = int(tokens[1])
            elif key == "device" and len(tokens) >= 2:
                self.device_type = tokens[1].upper()
                if len(tokens) >= 3 and tokens[2].isdigit():
                    self.device_index = int(tokens[2])
            elif key in ("cpu", "gpu", "tpu"):
                self.device_type = key.upper()
                if len(tokens) >= 2 and tokens[1].isdigit():
                    self.device_index = int(tokens[1])
        return self

    def to_string(self) -> str:
        """Return the canonical string representation of this `DeviceSpec`.

        Returns:
            str: Canonical device string.
        """
        segments: list[str] = []
        if self.job is not None:
            segments.append(f"/job:{self.job}")
        if self.replica is not None:
            segments.append(f"/replica:{self.replica}")
        if self.task is not None:
            segments.append(f"/task:{self.task}")
        if self.device_type is not None:
            idx_suffix = f":{self.device_index}" if self.device_index is not None else ":*"
            segments.append(f"/device:{self.device_type}{idx_suffix}")
        return "".join(segments)

    def make_merged_spec(self, dev: DeviceSpec) -> DeviceSpec:
        """Return a new `DeviceSpec` combining `self` with overrides from `dev`.

        Args:
            dev: Overriding `DeviceSpec`.

        Returns:
            DeviceSpec: Merged `DeviceSpec`.
        """
        return DeviceSpec(
            job=dev.job if dev.job is not None else self.job,
            replica=dev.replica if dev.replica is not None else self.replica,
            task=dev.task if dev.task is not None else self.task,
            device_type=dev.device_type if dev.device_type is not None else self.device_type,
            device_index=dev.device_index if dev.device_index is not None else self.device_index,
        )


class CriticalSection:
    """Execute a function mutually exclusively across threads."""

    def __init__(
        self,
        name: str | None = None,
        shared_name: str | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `CriticalSection`.

        Args:
            name: Optional critical section name.
            shared_name: Optional shared resource name.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self.name = name or shared_name or "critical_section"
        self._lock = threading.Lock()

    def execute(
        self,
        fn: Callable[[], _R],
        exclusive_resource_access: bool = True,
        name: str | None = None,
    ) -> _R:
        """Execute `fn()` while holding the critical section mutex.

        Args:
            fn: Zero-argument callable to execute under mutual exclusion.
            exclusive_resource_access: Whether resource access is exclusive.
            name: Optional operation name.

        Returns:
            _R: Return value of `fn()`.
        """
        del exclusive_resource_access, name
        with self._lock:
            return fn()


class VariableAggregation(str, enum.Enum):
    """Indicates how a distributed variable will be aggregated."""

    NONE = "none"
    SUM = "sum"
    MEAN = "mean"
    ONLY_FIRST_REPLICA = "only_first_replica"


class VariableSynchronization(str, enum.Enum):
    """Indicates when a distributed variable will be synced."""

    AUTO = "auto"
    NONE = "none"
    ON_WRITE = "on_write"
    ON_READ = "on_read"


class UnconnectedGradients(str, enum.Enum):
    """Controls how `GradientTape.gradient` handles unconnected target/source pairs."""

    NONE = "none"
    ZERO = "zero"


class AggregationMethod:
    """A class listing aggregation methods used to combine gradients."""

    ADD_N: int = 0
    DEFAULT: int = 0
    EXPERIMENTAL_TREE: int = 1
    EXPERIMENTAL_ACCUMULATE_N: int = 2


class RegisterGradient:
    """Decorator for registering the gradient function for an op type."""

    def __init__(self, op_type: str = "") -> None:
        """Initialize `RegisterGradient`.

        Args:
            op_type: String operation type for which to register a gradient.
        """
        self.op_type = op_type

    def __call__(self, f: GradFnProtocol) -> GradFnProtocol:
        """Register `f` as the gradient function for `self.op_type`.

        Args:
            f: Gradient callback function `(op, *grads) -> input_grads`.

        Returns:
            GradFnProtocol: Unmodified `f`.
        """
        _GRADIENT_REGISTRY[self.op_type] = f

        def _vjp_rule(
            graph: LogicalGraph,
            node: LogicalNode,
            cotangent: str,
            /,
        ) -> Sequence[str | _MscUnconnectedGradients | None] | str:
            """Evaluate registered TF gradient callback on IR `node`.

            Args:
                graph: Target IR graph.
                node: Target IR node.
                cotangent: Incoming cotangent node ID.

            Returns:
                Sequence[str | _MscUnconnectedGradients | None] | str: Cotangent IDs for inputs.
            """
            del graph
            f(node.op_type, cotangent)
            return [cotangent for _ in node.inputs]

        if self.op_type:
            _VJP_REGISTRY[self.op_type] = _vjp_rule
        return f


class Module(_ModuleTrackerProtocol):
    """Base neural network module class for tracking variables and submodules."""

    _direct_variables: list[Variable]
    _direct_submodules: list[Module]

    def __init_subclass__(cls) -> None:
        """Wrap subclass `__init__` so `_ACTIVE_MODULE_STACK` is popped on return."""
        super().__init_subclass__()
        orig_init = cls.__init__

        def _wrapped_init(
            self: Module,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Execute subclass `__init__` within an active module tracking scope."""
            if _ACTIVE_MODULE_STACK and isinstance(_ACTIVE_MODULE_STACK[-1], Module):
                _ACTIVE_MODULE_STACK[-1]._direct_submodules.append(self)
            _ACTIVE_MODULE_STACK.append(self)
            try:
                orig_init(self, *args, **kwargs)
            finally:
                _ACTIVE_MODULE_STACK.pop()

        cls.__init__ = _wrapped_init  # type: ignore[method-assign]

    def __new__(
        cls,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Module:
        """Allocate a `Module` instance with initialized tracking lists.

        Args:
            *args: Positional arguments forwarded to `__init__`.
            **kwargs: Keyword arguments forwarded to `__init__`.

        Returns:
            Module: Allocated `Module` instance.
        """
        del args, kwargs
        instance = super().__new__(cls)
        instance._direct_variables = []
        instance._direct_submodules = []
        return instance

    def __init__(
        self,
        name: str | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Module`.

        Args:
            name: Optional module name string.
            *args: Optional subclass positional arguments.
            **kwargs: Optional subclass keyword arguments.
        """
        del args, kwargs
        self.name = name if isinstance(name, str) and name else type(self).__name__.lower()

    def _register_variable(self, var: Variable) -> None:
        """Register a newly created `Variable` on this `Module`.

        Args:
            var: Newly created `Variable` instance.
        """
        self._direct_variables.append(var)

    @property
    def variables(self) -> builtins.tuple[Variable, ...]:
        """Return sequence of all `Variable` attributes owned by this module.

        Returns:
            tuple[Variable, ...]: Owned variables.
        """
        found: list[Variable] = list(self._direct_variables)
        for sub in self._direct_submodules:
            found.extend(sub.variables)
        return builtins.tuple(found)

    @property
    def trainable_variables(self) -> builtins.tuple[Variable, ...]:
        """Return sequence of trainable `Variable` attributes owned by this module.

        Returns:
            tuple[Variable, ...]: Trainable variables.
        """
        return builtins.tuple(v for v in self.variables if v.trainable)

    @property
    def non_trainable_variables(self) -> builtins.tuple[Variable, ...]:
        """Return sequence of non-trainable `Variable` attributes owned by this module.

        Returns:
            tuple[Variable, ...]: Non-trainable variables.
        """
        return builtins.tuple(v for v in self.variables if not v.trainable)

    @property
    def submodules(self) -> builtins.tuple[Module, ...]:
        """Return sequence of child `Module` instances owned by this module.

        Returns:
            tuple[Module, ...]: Child modules.
        """
        return builtins.tuple(self._direct_submodules)
