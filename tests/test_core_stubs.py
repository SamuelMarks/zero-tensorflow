"""Unit tests for `test_core_stubs`."""

from __future__ import annotations

from zero_tensorflow.core_stubs import (
    double,
    import_graph_def,
    init_scope,
    is_symbolic_tensor,
    is_tensor,
    ragged_fill_empty_rows,
    ragged_fill_empty_rows_grad,
    size,
    space_to_batch,
    space_to_batch_nd,
    string,
)

import ml_switcheroo_compiler.ops as msc_ops


def test_core_stubs() -> None:
    """Execute `test_core_stubs`."""
    tensor = msc_ops.zeros((2,))
    assert double(tensor) is not None
    import_graph_def()
    with init_scope():
        pass

    class SymbolicTensor:
        """Test helper class `SymbolicTensor`."""

        is_symbolic = True

    assert is_symbolic_tensor(SymbolicTensor()) is True
    assert is_symbolic_tensor(tensor) is False
    assert is_tensor(tensor) is True
    assert is_tensor(42) is False
    assert ragged_fill_empty_rows(1, 2) == (1, 2)
    assert ragged_fill_empty_rows_grad(1) == (1,)
    assert size(tensor) == 2
    assert size(42) == 0
    assert space_to_batch(tensor) is tensor
    assert space_to_batch() is None
    assert space_to_batch_nd(tensor) is tensor
    assert space_to_batch_nd() is None
    assert string(tensor) is not None
