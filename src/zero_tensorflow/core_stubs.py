"""Top-level stubs."""

from typing import Any

import ml_switcheroo_compiler.core as msc_core


def double(*args: Any, **kwargs: Any) -> Any:
    """Stub for double."""
    from zero_keras import ops as msc_ops

    return msc_ops.cast(args[0], msc_core.DType.Float64)


def import_graph_def(*args: Any, **kwargs: Any) -> Any:
    """Stub for import_graph_def."""
    return None


class init_scope:
    """Stub for init_scope context manager."""

    def __enter__(self) -> "init_scope":  # noqa: PYI034
        """Enter context."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context."""


def is_symbolic_tensor(tensor: Any) -> bool:
    """Stub for is_symbolic_tensor."""
    return getattr(tensor, "is_symbolic", False)


def is_tensor(tensor: Any) -> bool:
    """Check if object is a tensor."""
    return isinstance(
        tensor, (msc_core.Tensor, msc_core.SparseTensor, msc_core.RaggedTensor)
    )


def ragged_fill_empty_rows(*args: Any, **kwargs: Any) -> Any:
    """Stub for ragged_fill_empty_rows."""
    return args


def ragged_fill_empty_rows_grad(*args: Any, **kwargs: Any) -> Any:
    """Stub for ragged_fill_empty_rows_grad."""
    return args


def size(input: Any, out_type: Any = None, name: Any = None) -> Any:
    """Stub for size."""
    import math

    return math.prod(input.shape) if hasattr(input, "shape") and input.shape else 0


def space_to_batch(*args: Any, **kwargs: Any) -> Any:
    """Stub for space_to_batch."""
    return args[0] if args else None


def space_to_batch_nd(*args: Any, **kwargs: Any) -> Any:
    """Stub for space_to_batch_nd."""
    return args[0] if args else None


def string(*args: Any, **kwargs: Any) -> Any:
    """Stub for string."""
    return args[0] if args else None
