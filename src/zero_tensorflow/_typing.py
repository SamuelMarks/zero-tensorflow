"""Centralized domain type aliases and structural protocols."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.core.dtype as msc_dtype
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from typing_extensions import TypeAlias

ScalarNumeric: TypeAlias = "int | float | bool | complex"
ScalarValue: TypeAlias = "int | float | bool | complex | str | bytes | None"
NestedTensorData: TypeAlias = "ScalarValue | Sequence[NestedTensorData]"
ShapeTuple: TypeAlias = "tuple[int | None, ...]"
IntTuple: TypeAlias = "tuple[int, ...]"
ShapeLike: TypeAlias = "Sequence[int | None] | int | None"
DTypeLike: TypeAlias = (
    "msc_dtype.DType | str | type[int] | type[float] | type[bool] | type[complex] "
    "| type[str] | type[bytes] | None"
)
AxisLike: TypeAlias = "int | Sequence[int] | None"


@runtime_checkable
class ArrayLikeProtocol(Protocol):
    """Structural protocol for NumPy-like array objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array dimensions.

        Returns:
            tuple[int, ...]: Shape tuple.
        """

    @property
    def size(self) -> int:
        """Return total number of elements.

        Returns:
            int: Element count.
        """

    def tolist(self) -> NestedTensorData:
        """Convert array contents to nested Python lists or scalar.

        Returns:
            NestedTensorData: Nested Python list or scalar value.
        """

    def item(self) -> ScalarValue:
        """Extract single scalar value from a 0D or 1-element array.

        Returns:
            ScalarValue: Extracted Python scalar.
        """


@runtime_checkable
class NumpyConvertibleProtocol(Protocol):
    """Structural protocol for objects exposing a `.numpy()` conversion method."""

    def numpy(self) -> TensorOrScalar | ArrayLikeProtocol | NestedTensorData | TensorOperand:
        """Convert tensor value to a NumPy-like array or Python scalar/sequence.

        Returns:
            TensorOrScalar | ArrayLikeProtocol | NestedTensorData | TensorOperand: Materialized value.
        """


@runtime_checkable
class ValuesHolderProtocol(Protocol):
    """Structural protocol for sparse or composite tensors exposing a `.values` attribute."""

    @property
    def values(self) -> TensorOperand:
        """Return underlying values tensor or data.

        Returns:
            TensorOperand: Underlying values.
        """


@runtime_checkable
class AssignableProtocol(Protocol):
    """Structural protocol for mutable variable-like objects supporting `.assign()`."""

    def assign(self, value: TensorOperand) -> TensorOperand:
        """Assign a new value to the variable.

        Args:
            value: New value to assign.

        Returns:
            TensorOperand: Updated variable or tensor value.
        """


@runtime_checkable
class TrackableContainerProtocol(Protocol):
    """Structural protocol for containers (like models/layers) exposing `.variables`."""

    @property
    def variables(self) -> Sequence[AssignableProtocol]:
        """Return tracked variables owned by this container.

        Returns:
            Sequence[AssignableProtocol]: Sequence of tracked variables.
        """


@runtime_checkable
class CallableOperandProtocol(Protocol):
    """Structural protocol for callables passed as tensor or control-flow operands."""

    def __call__(
        self,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> TensorOperand:
        """Invoke the callable operand.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorOperand: Result of the callable.
        """


TensorLike: TypeAlias = (
    "msc_core.Tensor | ProxyTensor | ArrayLikeProtocol | NumpyConvertibleProtocol "
    "| ValuesHolderProtocol | AssignableProtocol | ScalarNumeric | Sequence[TensorLike]"
)

StringTensorLike: TypeAlias = (
    "msc_core.Tensor | ProxyTensor | ArrayLikeProtocol | NumpyConvertibleProtocol "
    "| str | bytes | Sequence[StringTensorLike]"
)

TrackableLike: TypeAlias = (
    "AssignableProtocol | TrackableContainerProtocol | NumpyConvertibleProtocol"
)

TensorOperand: TypeAlias = (
    "msc_core.Tensor | ProxyTensor | ArrayLikeProtocol | NumpyConvertibleProtocol "
    "| ValuesHolderProtocol | AssignableProtocol | ScalarValue "
    "| Sequence[TensorOperand] | Mapping[str, TensorOperand] | Iterable[TensorOperand]"
)

TwoTensorTuple: TypeAlias = "tuple[msc_core.Tensor, msc_core.Tensor]"


def as_operand(val: TensorOrScalar | EvalValue | TensorOperand) -> TensorOperand:
    """Narrow a `TensorOrScalar` or `EvalValue` to `TensorOperand`.

    Args:
        val: Candidate value returned by a compiler or array protocol method.

    Returns:
        TensorOperand: Narrowed operand value.

    Raises:
        TypeError: If `val` is not a valid `TensorOperand`.
    """
    if (
        isinstance(
            val,
            (
                msc_core.Tensor,
                ProxyTensor,
                ArrayLikeProtocol,
                NumpyConvertibleProtocol,
                ValuesHolderProtocol,
                AssignableProtocol,
                int,
                float,
                bool,
                complex,
                str,
                bytes,
            ),
        )
        and not isinstance(val, Mapping)
    ) or val is None:
        return val
    if isinstance(val, Sequence):
        return [as_operand(item) for item in val]
    raise TypeError(f"Unsupported tensor operand type: {type(val).__name__}")


__all__ = [
    "ArrayLikeProtocol",
    "AssignableProtocol",
    "AxisLike",
    "CallableOperandProtocol",
    "DTypeLike",
    "IntTuple",
    "NestedTensorData",
    "NumpyConvertibleProtocol",
    "ScalarNumeric",
    "ScalarValue",
    "ShapeLike",
    "ShapeTuple",
    "StringTensorLike",
    "TensorLike",
    "TensorOperand",
    "TrackableContainerProtocol",
    "TrackableLike",
    "TwoTensorTuple",
    "ValuesHolderProtocol",
    "as_operand",
]
