"""Stub module for optimizers."""

from zero_keras.optimizers import *  # noqa: F403
from typing import Any


class legacy:
    """Stub for legacy module."""

    class Adagrad:
        """Stub for Adagrad."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: Adagrad")

    class Adam:
        """Stub for Adam."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: Adam")

    class Ftrl:
        """Stub for Ftrl."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: Ftrl")

    class Optimizer:
        """Stub for Optimizer."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: Optimizer")

    class RMSprop:
        """Stub for RMSprop."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: RMSprop")

    SGD: int = 0
    """Stub for SGD."""
