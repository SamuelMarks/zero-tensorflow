"""TensorFlow lookup table module."""

from __future__ import annotations

from zero_tensorflow._typing import TensorOperand

__all__ = ["experimental"]


class experimental:
    """Experimental hash table lookup classes."""

    class DenseHashTable:
        """Mutable hash table using dense open addressing."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `DenseHashTable`.

            Args:
                *args: Positional table arguments.
                **kwargs: Keyword table arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class MutableHashTable:
        """Generic mutable hash table implementation."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `MutableHashTable`.

            Args:
                *args: Positional table arguments.
                **kwargs: Keyword table arguments.
            """
            self.args = args
            self.kwargs = kwargs
