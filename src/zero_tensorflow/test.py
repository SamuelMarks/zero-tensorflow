"""TensorFlow testing utilities module."""

from __future__ import annotations

from zero_tensorflow._typing import TensorOperand

__all__ = ["experimental"]


class experimental:
    """Experimental testing utilities."""

    @staticmethod
    def sync_devices(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Synchronize all active compute devices.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        return
