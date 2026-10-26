"""TensorFlow type annotations and experimental trace types module."""

from __future__ import annotations

import builtins
from collections.abc import Sequence

from zero_tensorflow._typing import TensorOperand

__all__ = ["experimental"]


class _BaseTraceType:
    """Base implementation for TensorFlow `TraceType` hierarchy."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize trace type descriptor.

        Args:
            *args: Positional trace type arguments.
            **kwargs: Keyword trace type arguments.
        """
        self.args = args
        self.kwargs = kwargs

    def is_subtype_of(self, other: builtins.object) -> bool:
        """Return whether `self` is a subtype of `other`.

        Args:
            other: Target trace type to check against.

        Returns:
            bool: `True` when `self` is an instance of `type(other)` and matches args.
        """
        if not isinstance(other, _BaseTraceType):
            return False
        if not isinstance(self, type(other)):
            return False
        return not (other.args and self.args != other.args)

    def most_specific_common_supertype(
        self,
        others: Sequence[_BaseTraceType],
    ) -> _BaseTraceType | None:
        """Return the most specific common supertype of `self` and `others`.

        Args:
            others: Sequence of other trace types.

        Returns:
            _BaseTraceType | None: Common supertype if compatible, else `None`.
        """
        if all(other.is_subtype_of(self) for other in others):
            return self
        return _BaseTraceType()


class _DistributedNamespace:
    """Distributed tensor type namespace (`tf.types.experimental.distributed`)."""

    class _PerReplica(_BaseTraceType):
        """Per-replica distributed value trace type."""

    class _Mirrored(_BaseTraceType):
        """Mirrored distributed value trace type."""

    def __call__(self, *args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
        """Allow invocation as a no-op stub for backwards compatibility.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorOperand: Always `None`.
        """
        del args, kwargs
        return None


class experimental:
    """Experimental TensorFlow trace and function types."""

    class TraceType(_BaseTraceType):
        """Base protocol for representing the type of objects for `tf.function` tracing."""

    class TensorLike(TraceType):
        """Base type for tensor-like objects (`Tensor`, `Variable`, `SparseTensor`)."""

    class SupportsTracingProtocol(TraceType):
        """Protocol for custom types supporting `tf.function` tracing."""

        def __tf_tracing_type__(self, context: TensorOperand = None) -> _BaseTraceType:
            """Return the `TraceType` for this object inside a tracing context.

            Args:
                context: Tracing context.

            Returns:
                _BaseTraceType: Trace type instance.
            """
            del context
            return experimental.TraceType(*self.args, **self.kwargs)

    class Callable(TraceType):
        """Base type for TensorFlow callables."""

    class FunctionType(Callable):
        """Represents the type signature of a TensorFlow function."""

    class AtomicFunction(Callable):
        """Base class for graph functions."""

    class ConcreteFunction(AtomicFunction):
        """A specialization of a `PolymorphicFunction` for a fixed input signature."""

    class GenericFunction(Callable):
        """Base class for polymorphic graph functions."""

    class PolymorphicFunction(GenericFunction):
        """Base class for `@tf.function` polymorphic callables."""

    distributed = _DistributedNamespace()
