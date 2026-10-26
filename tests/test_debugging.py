"""Unit tests for `zero_tensorflow.debugging`."""

from __future__ import annotations

import zero_tensorflow as tf
from zero_tensorflow import debugging

import pytest


class _FakeArray2D:
    """Pure-Python 2-D `ArrayLikeProtocol` test double."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return 2-D shape.

        Returns:
            tuple[int, ...]: Shape `(2, 3)`.
        """
        return (2, 3)

    @property
    def size(self) -> int:
        """Return element count.

        Returns:
            int: Element count `6`.
        """
        return 6

    def tolist(self) -> list[list[float]]:
        """Return nested list of floats.

        Returns:
            list[list[float]]: 2x3 zeroes list.
        """
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def item(self) -> float:
        """Return first element.

        Returns:
            float: `0.0`.
        """
        return 0.0


class _FakeArray1D:
    """Pure-Python 1-D `ArrayLikeProtocol` test double."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return 1-D shape.

        Returns:
            tuple[int, ...]: Shape `(2,)`.
        """
        return (2,)

    @property
    def size(self) -> int:
        """Return element count.

        Returns:
            int: Element count `2`.
        """
        return 2

    def tolist(self) -> list[float]:
        """Return 1-D list of floats.

        Returns:
            list[float]: `[1.0, 2.0]`.
        """
        return [1.0, 2.0]

    def item(self) -> float:
        """Return first element.

        Returns:
            float: `1.0`.
        """
        return 1.0


def test_debugging() -> None:
    """Verify `tf.debugging` assertion functions and experimental debug dump state."""
    assert debugging.experimental.enable_dump_debug_info("/tmp/tfdbg") is None
    assert debugging._DEBUG_DUMP_ROOT == "/tmp/tfdbg"
    assert debugging.experimental.enable_dump_debug_info(None) is None
    assert debugging.experimental.disable_dump_debug_info() is None

    debugging.assert_equal(None, None)
    debugging.assert_equal(tf.Tensor([1.0, 2.0]), [1.0, 2.0])
    with pytest.raises(ValueError, match="assert_equal"):
        debugging.assert_equal([1.0, 2.0], [1.0, 3.0])

    debugging.assert_near(None, None)
    debugging.assert_near([1.0, 2.0], [1.0 + 1e-7, 2.0 - 1e-7])
    with pytest.raises(ValueError, match="assert_near"):
        debugging.assert_near([1.0], [2.0])

    debugging.assert_less(None, None)
    debugging.assert_less([1.0, 2.0], [3.0, 4.0])
    with pytest.raises(ValueError, match="assert_less"):
        debugging.assert_less([3.0], [2.0])

    debugging.assert_greater(None, None)
    debugging.assert_greater([5.0, 6.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="assert_greater"):
        debugging.assert_greater([1.0], [2.0])

    debugging.assert_non_negative(None)
    debugging.assert_non_negative([0.0, 1.0])
    with pytest.raises(ValueError, match="assert_non_negative"):
        debugging.assert_non_negative([-0.1])

    debugging.assert_positive(None)
    debugging.assert_positive([0.1, 1.0])
    with pytest.raises(ValueError, match="assert_positive"):
        debugging.assert_positive([0.0])

    debugging.assert_proper_iterable([1, 2, 3])
    with pytest.raises(TypeError, match="proper iterable"):
        debugging.assert_proper_iterable(tf.Tensor([1, 2]))
    with pytest.raises(TypeError, match="proper iterable"):
        debugging.assert_proper_iterable("hello")

    debugging.assert_rank(None, None)
    debugging.assert_rank(tf.Tensor([[1.0, 2.0]]), 2)
    debugging.assert_rank(_FakeArray2D(), 2)
    debugging.assert_rank([], 1)
    with pytest.raises(ValueError, match="Expected rank"):
        debugging.assert_rank([1.0, 2.0], 2)

    t_a = tf.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t_b = tf.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    debugging.assert_shapes({t_a.ref(): ("N", "M"), t_b.ref(): ("M", None)})
    debugging.assert_shapes([(t_a, (2, 3))])
    with pytest.raises(ValueError, match="Rank mismatch"):
        debugging.assert_shapes([(t_a, (2,))])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        debugging.assert_shapes([(t_a, (2, 4))])
    with pytest.raises(ValueError, match="Symbolic dimension"):
        debugging.assert_shapes([(t_a, ("N", "N"))])

    debugging.assert_type(t_a, tf.float64)
    with pytest.raises(TypeError, match="Expected dtype"):
        debugging.assert_type(t_a, tf.int32)

    valid_t = debugging.check_numerics(tf.Tensor([1.0, 2.0]), "valid")
    assert isinstance(valid_t, tf.Tensor)
    with pytest.raises(ValueError, match="NaN or Inf"):
        debugging.check_numerics([1.0, float("nan")], "bad_nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        debugging.check_numerics([1.0, float("inf")], "bad_inf")

    assert debugging.is_numeric_tensor(t_a) is True
    assert debugging.is_numeric_tensor(tf.Tensor(True)) is False
    assert debugging.is_numeric_tensor("not_a_tensor") is False

    class _CustomNonNumericTensor(tf.Tensor):
        """Helper tensor returning a non-numeric dtype."""

        @property
        def dtype(self) -> None:
            """Return None dtype."""
            return None

    assert debugging.is_numeric_tensor(_CustomNonNumericTensor(1.0)) is False
    assert debugging._flatten_floats("non_num") == []
    assert debugging._flatten_floats(_FakeArray1D()) == [1.0, 2.0]
    assert debugging._infer_shape(_FakeArray1D()) == (2,)

    class _NumpyOnly:
        """Helper implementing `NumpyConvertibleProtocol`."""

        def numpy(self) -> list[float]:
            """Return list of floats."""
            return [1.0, 2.0]

    assert debugging._infer_shape(_NumpyOnly()) == (2,)
