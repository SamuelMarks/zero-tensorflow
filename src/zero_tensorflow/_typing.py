"""Centralized domain type aliases and structural protocols."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, TypeVar, runtime_checkable

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.core.dtype as msc_dtype
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_ir import LogicalNode
from typing_extensions import TypeAlias

_SeqItemT = TypeVar("_SeqItemT")
NestedSequence: TypeAlias = "_SeqItemT | Sequence[NestedSequence[_SeqItemT]]"
ScalarNumeric: TypeAlias = "int | float | bool | complex"
NumericScalar: TypeAlias = "int | float | bool | complex"
ScalarTensorValue: TypeAlias = "int | float | bool | complex | str | bytes"
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
KernelAttrValue: TypeAlias = (
    "ScalarTensorValue | Sequence[ScalarTensorValue] | DTypeLike | ShapeLike | None"
)


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
class GradFnProtocol(Protocol):
    """Structural protocol for backward gradient callbacks."""

    def __call__(
        self,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> TensorOperand:
        """Compute backward gradients given upstream gradients.

        Args:
            *args: Upstream gradient tensor(s).
            **kwargs: Optional keyword arguments (e.g., `variables`).

        Returns:
            TensorOperand: Input gradient(s) or `(input_grads, variable_grads)`.
        """


@runtime_checkable
class CustomGradientFnProtocol(Protocol):
    """Structural protocol for `@tf.custom_gradient` decorated forward functions."""

    def __call__(
        self,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> tuple[TensorOperand, GradFnProtocol]:
        """Compute forward primal result and backward gradient callback.

        Args:
            *args: Forward positional arguments.
            **kwargs: Forward keyword arguments.

        Returns:
            tuple[TensorOperand, GradFnProtocol]: Pair of `(primal_output, grad_fn)`.
        """


@runtime_checkable
class DatasetMapFnProtocol(Protocol):
    """Structural protocol for dataset element transformation callables."""

    def __call__(self, *args: TensorOperand) -> TensorOperand:
        """Transform a dataset element.

        Args:
            *args: Dataset element components.

        Returns:
            TensorOperand: Transformed dataset element.
        """


@runtime_checkable
class CondBranchProtocol(Protocol):
    """Structural protocol for zero-argument `tf.cond` / `tf.case` branch callables."""

    def __call__(self) -> TensorOperand:
        """Evaluate the conditional branch.

        Returns:
            TensorOperand: Branch result tensor or structure.
        """


@runtime_checkable
class WhileCondProtocol(Protocol):
    """Structural protocol for `tf.while_loop` condition callables."""

    def __call__(self, *args: TensorOperand) -> TensorOperand:
        """Evaluate loop continuation condition on current loop state.

        Args:
            *args: Current loop state variables.

        Returns:
            TensorOperand: Boolean scalar or tensor condition result.
        """


@runtime_checkable
class WhileBodyProtocol(Protocol):
    """Structural protocol for `tf.while_loop` body step callables."""

    def __call__(self, *args: TensorOperand) -> TensorOperand:
        """Execute one loop body step on current loop state.

        Args:
            *args: Current loop state variables.

        Returns:
            TensorOperand: Updated loop state variables.
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


@runtime_checkable
class TensorSpecProtocol(Protocol):
    """Structural protocol for `TensorSpec` and `TypeSpec` descriptors."""

    @property
    def shape(self) -> ShapeTuple:
        """Return expected tensor shape tuple.

        Returns:
            ShapeTuple: Dimension tuple.
        """

    @property
    def dtype(self) -> msc_dtype.DType | str:
        """Return expected tensor data type.

        Returns:
            msc_dtype.DType | str: Expected `DType`.
        """

    def is_compatible_with(self, spec_or_tensor: TensorOperand) -> bool:
        """Check whether `spec_or_tensor` is compatible with this specification.

        Args:
            spec_or_tensor: Candidate spec or tensor.

        Returns:
            bool: `True` if compatible.
        """


@runtime_checkable
class RaggedTensorProtocol(Protocol):
    """Structural protocol for `RaggedTensor` objects."""

    @property
    def flat_values(self) -> TensorOperand:
        """Return flattened innermost values.

        Returns:
            TensorOperand: Flat values.
        """

    @property
    def row_splits(self) -> tuple[int, ...]:
        """Return outermost row split offsets.

        Returns:
            tuple[int, ...]: Row split offsets.
        """


@runtime_checkable
class SparseTensorProtocol(Protocol):
    """Structural protocol for `SparseTensor` objects."""

    @property
    def indices(self) -> TensorOperand:
        """Return non-zero coordinate indices.

        Returns:
            TensorOperand: Coordinate indices.
        """

    @property
    def values(self) -> TensorOperand:
        """Return non-zero values vector.

        Returns:
            TensorOperand: Non-zero values.
        """

    @property
    def dense_shape(self) -> TensorOperand:
        """Return full dense tensor shape.

        Returns:
            TensorOperand: Dense shape.
        """


@runtime_checkable
class TrackableProtocol(Protocol):
    """Structural protocol for checkpoint-trackable objects."""

    @property
    def name(self) -> str:
        """Return trackable resource identifier.

        Returns:
            str: Resource name.
        """


EagerTensorOperand: TypeAlias = (
    "msc_core.Tensor | ArrayLikeProtocol | NumpyConvertibleProtocol "
    "| ScalarValue | Sequence[EagerTensorOperand]"
)

SymbolicTensorOperand: TypeAlias = "ProxyTensor | LogicalNode"

TensorLike: TypeAlias = (
    "msc_core.Tensor | ProxyTensor | ArrayLikeProtocol | NumpyConvertibleProtocol "
    "| ValuesHolderProtocol | AssignableProtocol | ScalarNumeric | Sequence[TensorLike]"
)

TensorCompatible: TypeAlias = (
    "msc_core.Tensor | ProxyTensor | msc_dtype.DType | ArrayLikeProtocol "
    "| NumpyConvertibleProtocol | ValuesHolderProtocol | AssignableProtocol "
    "| ScalarTensorValue | Sequence[TensorCompatible]"
)

NestedTensorStructure: TypeAlias = (
    "TensorCompatible | Sequence[NestedTensorStructure] "
    "| Mapping[str, NestedTensorStructure] | None"
)

StringTensorLike: TypeAlias = (
    "msc_core.Tensor | ProxyTensor | ArrayLikeProtocol | NumpyConvertibleProtocol "
    "| str | bytes | Sequence[StringTensorLike]"
)

TrackableLike: TypeAlias = (
    "AssignableProtocol | TrackableContainerProtocol | TrackableProtocol | NumpyConvertibleProtocol"
)

TensorOperand: TypeAlias = (
    "msc_core.Tensor | ProxyTensor | ArrayLikeProtocol | NumpyConvertibleProtocol "
    "| ValuesHolderProtocol | AssignableProtocol | TensorSpecProtocol "
    "| RaggedTensorProtocol | SparseTensorProtocol | ScalarValue "
    "| Sequence[TensorOperand] | Mapping[str, TensorOperand] | Iterable[TensorOperand]"
)

TwoTensorTuple: TypeAlias = "tuple[msc_core.Tensor, msc_core.Tensor]"


def ensure_tensor_compatible(
    val: TensorOrScalar | EvalValue | TensorOperand,
) -> TensorCompatible:
    """Narrow a candidate value to `TensorCompatible`.

    Args:
        val: Candidate value to validate.

    Returns:
        TensorCompatible: Validated tensor-compatible value.

    Raises:
        TypeError: If `val` cannot be represented as `TensorCompatible`.
    """
    if isinstance(
        val,
        (
            msc_core.Tensor,
            ProxyTensor,
            msc_dtype.DType,
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
    ) and not isinstance(val, Mapping):
        return val
    if isinstance(val, Sequence):
        return [ensure_tensor_compatible(item) for item in val]
    raise TypeError(f"Value of type {type(val).__name__} is not TensorCompatible")


def ensure_scalar(val: ScalarValue | TensorOperand) -> ScalarTensorValue:
    """Narrow a candidate scalar value to `ScalarTensorValue`.

    Args:
        val: Candidate scalar value.

    Returns:
        ScalarTensorValue: Narrowed Python scalar (`int`, `float`, `bool`, `complex`, `str`, `bytes`).

    Raises:
        TypeError: If `val` is not a primitive scalar value.
    """
    if isinstance(val, (int, float, bool, complex, str, bytes)):
        return val
    raise TypeError(f"Expected scalar value, got {type(val).__name__}")


def ensure_shape_tuple(shape: ShapeLike | TensorOperand) -> ShapeTuple:
    """Convert a `ShapeLike` value into a normalized `ShapeTuple`.

    Args:
        shape: Shape integer, sequence of optional dimension sizes, or `None`.

    Returns:
        ShapeTuple: Normalized tuple of `int | None` dimensions.
    """
    if shape is None:
        return ()
    if isinstance(shape, int) and not isinstance(shape, bool):
        return (shape,)
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        return tuple(
            int(d) if isinstance(d, int) and not isinstance(d, bool) else None for d in shape
        )
    return ()


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
                TensorSpecProtocol,
                RaggedTensorProtocol,
                SparseTensorProtocol,
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
    "CondBranchProtocol",
    "CustomGradientFnProtocol",
    "DTypeLike",
    "DatasetMapFnProtocol",
    "EagerTensorOperand",
    "GradFnProtocol",
    "IntTuple",
    "KernelAttrValue",
    "NestedSequence",
    "NestedTensorData",
    "NestedTensorStructure",
    "NumericScalar",
    "NumpyConvertibleProtocol",
    "RaggedTensorProtocol",
    "ScalarNumeric",
    "ScalarTensorValue",
    "ScalarValue",
    "ShapeLike",
    "ShapeTuple",
    "SparseTensorProtocol",
    "StringTensorLike",
    "SymbolicTensorOperand",
    "TensorCompatible",
    "TensorLike",
    "TensorOperand",
    "TensorSpecProtocol",
    "TrackableContainerProtocol",
    "TrackableLike",
    "TrackableProtocol",
    "TwoTensorTuple",
    "ValuesHolderProtocol",
    "WhileBodyProtocol",
    "WhileCondProtocol",
    "as_operand",
    "ensure_scalar",
    "ensure_shape_tuple",
    "ensure_tensor_compatible",
]
