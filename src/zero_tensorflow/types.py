"""TensorFlow type annotations and experimental trace types module."""

from __future__ import annotations

from zero_tensorflow._typing import TensorOperand

__all__ = ["experimental"]


class experimental:
    """Experimental function and tensor typing protocols."""

    class AtomicFunction:
        """Represents a traced atomic graph function."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `AtomicFunction`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class Callable:
        """Base type for TensorFlow callables."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `Callable`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class ConcreteFunction:
        """Base type for specialized monomorphic graph functions."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `ConcreteFunction`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class FunctionType:
        """Represents the type signature of a TensorFlow function."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `FunctionType`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class GenericFunction:
        """Base type for polymorphic TensorFlow functions."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `GenericFunction`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class PolymorphicFunction:
        """Deprecated alias for `GenericFunction`."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `PolymorphicFunction`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class SupportsTracingProtocol:
        """Protocol for objects that can be traced into a `TraceType`."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `SupportsTracingProtocol`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class TensorLike:
        """Base type for objects convertible to `Tensor`."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `TensorLike`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class TraceType:
        """Represents the type of object(s) for `tf.function` tracing."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `TraceType`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    @staticmethod
    def distributed(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Distributed tensor type namespace stub.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        return
