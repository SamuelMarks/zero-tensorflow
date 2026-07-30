"""Stub module for summary."""

import contextlib
from typing import Any

_SUMMARY_STEP = None


class experimental:
    """Stub for experimental module."""

    @staticmethod
    def get_step() -> Any:
        """Stub for get_step."""
        return _SUMMARY_STEP

    @staticmethod
    def set_step(step: Any) -> None:
        """Stub for set_step."""
        global _SUMMARY_STEP
        _SUMMARY_STEP = step

    @staticmethod
    @contextlib.contextmanager
    def summary_scope(*args: Any, **kwargs: Any) -> Any:
        """Stub for summary_scope."""
        yield ("summary_scope", None)

    @staticmethod
    def write_raw_pb(*args: Any, **kwargs: Any) -> None:
        """Stub for write_raw_pb."""


def image(*args: Any, **kwargs: Any) -> None:
    """Stub for image."""
