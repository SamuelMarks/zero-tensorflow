"""TensorFlow sparse module."""

from typing import Any, Optional, Sequence

__all__ = [
    "SparseTensor",
    "sparse_dense_matmul",
    "add",
    "concat",
    "split",
    "to_dense",
    "from_dense",
]


class SparseTensor:
    """
    Represents a sparse tensor.

    Args:
        indices: A 2-D int64 tensor of shape `[N, ndims]`.
        values: A 1-D tensor of any type and shape `[N]`.
        dense_shape: A 1-D int64 tensor of shape `[ndims]`.
    """

    def __init__(self, indices: Any, values: Any, dense_shape: Any):
        """Initialize the object."""
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape


def sparse_dense_matmul(
    sp_a: Any,
    b: Any,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    name: Optional[str] = None,
) -> Any:
    """
    Multiply SparseTensor by dense matrix.

    Args:
        sp_a: SparseTensor A.
        b: dense Tensor B.
        adjoint_a: Use the adjoint of A in the matrix multiply.
        adjoint_b: Use the adjoint of B in the matrix multiply.
        name: A name prefix for the returned tensors (optional).
    """
    raise NotImplementedError("Not implemented: tf.sparse.sparse_dense_matmul")


def add(a: Any, b: Any, threshold: int = 0, name: Optional[str] = None) -> Any:
    """
    Add two tensors, at least one of each is a SparseTensor.

    Args:
        a: The first operand; SparseTensor or Tensor.
        b: The second operand; SparseTensor or Tensor.
        threshold: An optional 0-D int32 Tensor.
        name: A name prefix for the returned tensors (optional).
    """
    raise NotImplementedError("Not implemented: tf.sparse.add")


def concat(axis: int, sp_inputs: Sequence[Any], name: Optional[str] = None) -> Any:
    """
    Concatenate a list of SparseTensor along the specified dimension.

    Args:
        axis: Dimension to concatenate along.
        sp_inputs: List of SparseTensor to concatenate.
        name: A name prefix for the returned tensors (optional).
    """
    raise NotImplementedError("Not implemented: tf.sparse.concat")


def split(sp_input: Any, num_split: int, axis: int, name: Optional[str] = None) -> Any:
    """
    Split a SparseTensor into num_split tensors along axis.

    Args:
        sp_input: The SparseTensor to split.
        num_split: A Python integer.
        axis: A 0-D int32 Tensor.
        name: A name prefix for the returned tensors (optional).
    """
    raise NotImplementedError("Not implemented: tf.sparse.split")


def to_dense(
    sp_input: Any,
    default_value: Optional[Any] = None,
    validate_indices: bool = True,
    name: Optional[str] = None,
) -> Any:
    """
    Convert a SparseTensor into a dense tensor.

    Args:
        sp_input: The input SparseTensor.
        default_value: Scalar value to set for indices not specified in sp_input.
        validate_indices: A boolean value.
        name: A name prefix for the returned tensors (optional).
    """
    raise NotImplementedError("Not implemented: tf.sparse.to_dense")


def from_dense(tensor: Any, name: Optional[str] = None) -> SparseTensor:
    """
    Convert a dense tensor into a SparseTensor.

    Args:
        tensor: A dense Tensor to be converted to a SparseTensor.
        name: Optional name for the op.
    """
    raise NotImplementedError("Not implemented: tf.sparse.from_dense")
