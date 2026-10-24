"""TensorFlow AutoGraph module."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from zero_tensorflow._typing import TensorOperand

from ml_switcheroo_compiler.tracing.autograph import do_not_convert, set_loop_options

F = TypeVar("F")

__all__ = ["experimental"]


class experimental:
    """Experimental AutoGraph APIs."""

    class Feature:
        """AutoGraph conversion feature flags."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `Feature`.

            Args:
                *args: Positional feature arguments.
                **kwargs: Keyword feature arguments.
            """
            self.args = args
            self.kwargs = kwargs

    @staticmethod
    def do_not_convert(
        func: F | None = None,
    ) -> F | Callable[[F], F]:
        """Mark a function as excluded from AutoGraph conversion.

        Args:
            func: Optional target function to decorate.

        Returns:
            F | Callable[[F], F]: Decorated function or decorator callable.
        """
        return do_not_convert(func)

    @staticmethod
    def set_loop_options(
        parallel_iterations: int | None = None,
        swap_memory: bool | None = None,
        maximum_iterations: int | None = None,
        shape_invariants: tuple[int | None, ...] | None = None,
    ) -> None:
        """Specify loop options for the enclosing loop in AutoGraph.

        Args:
            parallel_iterations: Maximum number of parallel loop iterations.
            swap_memory: Whether to enable memory swapping.
            maximum_iterations: Hard limit on loop iterations.
            shape_invariants: Shape invariants for loop variables.
        """
        set_loop_options(
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            maximum_iterations=maximum_iterations,
            shape_invariants=shape_invariants,
        )
