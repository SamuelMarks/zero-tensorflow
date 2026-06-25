"""TensorFlow ragged module."""

from typing import Any, Optional

__all__ = [
    "boolean_mask",
    "map_flat_values",
]


class RaggedTensor:
    """Represents a ragged tensor."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the object."""
        self._args = args
        self._kwargs = kwargs


def boolean_mask(data: Any, mask: Any, name: Optional[str] = None) -> Any:
    """
    Apply a boolean mask to data without flattening the mask dimensions.

    Args:
        data: A potentially ragged tensor.
        mask: A potentially ragged boolean tensor.
        name: A name prefix for the returned tensors (optional).
    """
    raise NotImplementedError("Not implemented: tf.ragged.boolean_mask")


def map_flat_values(op: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Apply op to the flat_values of one or more RaggedTensors.

    Args:
        op: The operation that should be applied to the RaggedTensor flat_values.
        *args: Arguments for op.
        **kwargs: Keyword arguments for op.
    """
    raise NotImplementedError("Not implemented: tf.ragged.map_flat_values")
