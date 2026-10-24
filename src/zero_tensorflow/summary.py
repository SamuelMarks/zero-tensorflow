"""TensorFlow TensorBoard summary logging module."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from zero_tensorflow._typing import TensorOperand

_SUMMARY_STEP: int | TensorOperand | None = None

__all__ = ["experimental", "image"]


class experimental:
    """Experimental summary step and scope utilities."""

    @staticmethod
    def get_step() -> int | TensorOperand | None:
        """Return the current default summary step for the thread.

        Returns:
            int | TensorOperand | None: Current summary step value or None.
        """
        return _SUMMARY_STEP

    @staticmethod
    def set_step(step: int | TensorOperand | None = 0) -> None:
        """Set the default summary step for the current thread.

        Args:
            step: An `int64`-convertible scalar step value.
        """
        global _SUMMARY_STEP
        _SUMMARY_STEP = step

    @staticmethod
    @contextlib.contextmanager
    def summary_scope(*args: TensorOperand, **kwargs: TensorOperand) -> Iterator[tuple[str, None]]:
        """Context manager for grouping summary operations under a name scope.

        Args:
            *args: Positional scope arguments.
            **kwargs: Keyword scope arguments.

        Yields:
            tuple[str, None]: Tuple of `(scope_name, tag_metadata)`.
        """
        del args, kwargs
        yield ("summary_scope", None)

    @staticmethod
    def write_raw_pb(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Write raw serialized `Summary` protocol buffers.

        Args:
            *args: Positional serialized protobuf arguments.
            **kwargs: Keyword summary arguments.
        """
        del args, kwargs


def image(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Write an image summary.

    Args:
        *args: Positional image summary arguments.
        **kwargs: Keyword image summary arguments.
    """
    del args, kwargs
