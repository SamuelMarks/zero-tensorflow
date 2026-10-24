"""TensorFlow debugging utilities module."""

from __future__ import annotations

from zero_tensorflow._typing import TensorOperand

__all__ = ["experimental"]


class experimental:
    """Experimental debugging APIs."""

    @staticmethod
    def disable_dump_debug_info(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Disable dumping of execution debug information.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        return

    @staticmethod
    def enable_dump_debug_info(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Enable dumping of execution debug information to disk.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        return
