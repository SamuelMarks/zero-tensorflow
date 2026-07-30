"""Stub module for autograph."""

from typing import Any

from ml_switcheroo_compiler.tracing.autograph import do_not_convert, set_loop_options


class experimental:
    """Stub for experimental module."""

    class Feature:
        """Stub for Feature."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    @staticmethod
    def do_not_convert(func: Any = None) -> Any:
        """do_not_convert."""
        return do_not_convert(func)

    @staticmethod
    def set_loop_options(*args: Any, **kwargs: Any) -> None:
        """set_loop_options."""
        return set_loop_options(*args, **kwargs)
