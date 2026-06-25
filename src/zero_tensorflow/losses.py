"""Stub module for losses."""

from zero_keras.losses import *  # noqa: F403
from typing import Any


KLD: int = 0
"""Stub for KLD."""

MAE: int = 0
"""Stub for MAE."""

MAPE: int = 0
"""Stub for MAPE."""

MSE: int = 0
"""Stub for MSE."""

MSLE: int = 0
"""Stub for MSLE."""


class Reduction:
    """Stub for Reduction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: Reduction")


def kld(*args: Any, **kwargs: Any) -> None:
    """Stub for kld."""
    raise NotImplementedError("Not implemented: kld")


def kullback_leibler_divergence(*args: Any, **kwargs: Any) -> None:
    """Stub for kullback_leibler_divergence."""
    raise NotImplementedError("Not implemented: kullback_leibler_divergence")


def logcosh(*args: Any, **kwargs: Any) -> None:
    """Stub for logcosh."""
    raise NotImplementedError("Not implemented: logcosh")


def mae(*args: Any, **kwargs: Any) -> None:
    """Stub for mae."""
    raise NotImplementedError("Not implemented: mae")


def mape(*args: Any, **kwargs: Any) -> None:
    """Stub for mape."""
    raise NotImplementedError("Not implemented: mape")


def mse(*args: Any, **kwargs: Any) -> None:
    """Stub for mse."""
    raise NotImplementedError("Not implemented: mse")


def msle(*args: Any, **kwargs: Any) -> None:
    """Stub for msle."""
    raise NotImplementedError("Not implemented: msle")
