"""Unit tests for `zero_tensorflow.test`."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import zero_tensorflow as tf
from zero_tensorflow import test

import pytest


def test_test() -> None:
    """Verify `tf.test.TestCase`, GPU queries, `main`, and `experimental.sync_devices`."""
    assert test.experimental.sync_devices() is None
    assert test.is_built_with_cuda() is False
    assert test.is_built_with_gpu_support() is False
    assert test.is_built_with_rocm() is False
    assert test.is_built_with_xla() is False
    assert test.gpu_device_name() == ""

    tc = test.TestCase()
    assert tc.evaluate(tf.Tensor(5.0)) == 5.0
    assert tc.evaluate((tf.Tensor(1.0), [tf.Variable(2.0), tf.Tensor(3.0)], [4.0])) == (
        1.0,
        [2.0, 3.0],
        [4.0],
    )

    class _NumpyObj:
        """Helper implementing `NumpyConvertibleProtocol`."""

        def numpy(self) -> float:
            """Return float."""
            return 4.0

    class _ListObj:
        """Helper implementing `ArrayLikeProtocol`."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Return shape."""
            return (2,)

        @property
        def size(self) -> int:
            """Return element count."""
            return 2

        def tolist(self) -> list[float]:
            """Return list."""
            return [1.0, 2.0]

        def item(self) -> float:
            """Return first element."""
            return 1.0

    assert tc.evaluate(_NumpyObj()) == 4.0
    assert test._shape_of(_NumpyObj()) == ()
    assert test._shape_of(_ListObj()) == (2,)
    assert test._shape_of([1.0, 2.0]) == (2,)
    assert test._shape_of([]) == (0,)

    tc.assertAllClose([1.0, 2.0], [1.0 + 1e-7, 2.0 - 1e-7])
    with pytest.raises(AssertionError):
        tc.assertAllClose([1.0], [2.0])

    tc.assertNotAllClose([1.0], [2.0])
    with pytest.raises(AssertionError):
        tc.assertNotAllClose([1.0], [1.0])

    tc.assertAllEqual([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(AssertionError):
        tc.assertAllEqual([1.0], [2.0])

    tc.assertNotAllEqual([1.0], [2.0])
    with pytest.raises(AssertionError):
        tc.assertNotAllEqual([1.0], [1.0])

    tc.assertAllGreater([2.0, 3.0], 1.0)
    with pytest.raises(AssertionError):
        tc.assertAllGreater([0.5, 3.0], 1.0)

    tc.assertAllLess([0.1, 0.2], 1.0)
    with pytest.raises(AssertionError):
        tc.assertAllLess([0.1, 2.0], 1.0)

    tc.assertShapeEqual([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], tf.zeros((2, 3)))
    with pytest.raises(AssertionError):
        tc.assertShapeEqual([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], tf.zeros((3, 2)))

    with tc.cached_session() as sess1:
        assert sess1 is tc
    with tc.session() as sess2:
        assert sess2 is tc
    tmp_d = tc.get_temp_dir()
    assert Path(tmp_d).exists()

    with mock.patch("unittest.main") as mock_main:
        test.main(["prog"])
        mock_main.assert_called_once()
