"""Unit tests for `test_extra_coverage`."""

from __future__ import annotations

import zero_tensorflow as tf
from zero_tensorflow import GradientTape, Tensor, _to_tensor, to_dtype
from zero_tensorflow.nn import softmax

import numpy as np
import pytest
from ml_switcheroo_compiler.core.dtype import DType


def test_tensor_numpy_and_comparisons() -> None:
    """Verify `Tensor` arithmetic, comparisons, boolean conversion, and `len()`."""
    t1 = Tensor([1.0, 2.0])
    t2 = Tensor([1.0, 3.0])

    class _NonOperand:
        """Unrelated sentinel class not convertible to a `TensorOperand`."""

    eq = tf.math.equal(t1, t2)
    assert isinstance(t1 == t2, (Tensor, bool))
    np.testing.assert_array_equal(eq.numpy(), [True, False])
    eq_non = t1 == _NonOperand()
    assert eq_non is False

    ne = tf.math.not_equal(t1, t2)
    assert isinstance(t1 != t2, (Tensor, bool))
    np.testing.assert_array_equal(ne.numpy(), [False, True])
    ne_non = t1 != _NonOperand()
    assert ne_non is True

    assert isinstance(t1 < t2, Tensor)
    assert isinstance(t1 <= t2, Tensor)
    assert isinstance(t1 > t2, Tensor)
    assert isinstance(t1 >= t2, Tensor)

    assert len(t1) == 2
    with pytest.raises(TypeError, match=r"Scalar tensor has no `len\(\)`"):
        len(Tensor(1.0))

    assert bool(Tensor(1.0)) is True
    assert bool(Tensor(0.0)) is False
    assert Tensor(1.0).__nonzero__() is True
    with pytest.raises(ValueError, match="ambiguous"):
        bool(t1)


def test_gradient_tape_unwatched_and_errors() -> None:
    """Verify `GradientTape` error handling and unwatched source behavior."""
    x = Tensor(2.0)
    unwatched = Tensor(5.0)
    with GradientTape() as tape:
        tape.watch(x)
        y = x * 3.0

    grad_unwatched = tape.gradient(y, unwatched)
    assert grad_unwatched is None

    with pytest.raises(RuntimeError, match="non-persistent"):
        tape.gradient(y, x)

    with GradientTape(persistent=True) as ptape:
        ptape.watch(x)
        y2 = x * 4.0

    g1 = ptape.gradient(y2, x)
    g2 = ptape.gradient(y2, x, unconnected_gradients="zero")
    g_unconn = ptape.gradient(y2, unwatched, unconnected_gradients="zero")
    assert isinstance(g1, Tensor) and g1.numpy() == 4.0
    assert isinstance(g2, Tensor) and g2.numpy() == 4.0
    assert isinstance(g_unconn, Tensor) and g_unconn.numpy() == 0.0

    with pytest.raises(ValueError, match="Unknown value for unconnected_gradients"):
        ptape.gradient(y2, x, unconnected_gradients="invalid")

    with pytest.raises(ValueError, match="not created during the tape"):
        ptape.gradient(unwatched, x)

    assert ptape.gradient(None, x) is None


def test_softmax_axis() -> None:
    """Verify `nn.softmax` numerical accuracy along default and explicit axes."""
    logits = Tensor([[1.0, 1.0], [0.0, 0.0]])
    probs = softmax(logits, axis=-1)
    np.testing.assert_allclose(probs.numpy(), [[0.5, 0.5], [0.5, 0.5]])


def test_to_tensor_invalid_dtype() -> None:
    """Verify `to_dtype` mappings for built-in types and `TypeError` on invalid dtypes."""
    assert to_dtype(None) == DType.Float32
    assert to_dtype(DType.Int64) == DType.Int64
    assert to_dtype("float64") == DType.Float64
    assert to_dtype("INT32") == DType.Int32
    assert to_dtype(bool) == DType.Bool
    assert to_dtype(int) == DType.Int32
    assert to_dtype(float) == DType.Float32
    assert to_dtype(complex) == DType.Complex64
    assert to_dtype(str) == DType.String
    assert to_dtype(bytes) == DType.String
    assert to_dtype(np.int16) == DType.Int16

    with pytest.raises(TypeError, match="Cannot convert"):
        to_dtype("not_a_valid_dtype")

    with pytest.raises(TypeError, match="Cannot convert"):
        to_dtype([1, 2, 3])


def test_compiler_ops_missing_coverage() -> None:
    """Verify `_to_tensor` and `Variable` edge cases (`read_value`, `name`, `dtype`)."""
    v = tf.Variable([1.0, 2.0], name="weights", dtype="float32")
    assert v.name == "weights"
    assert v.dtype == DType.Float32
    snap = v.read_value()
    assert isinstance(snap, Tensor)
    np.testing.assert_allclose(snap.numpy(), [1.0, 2.0])
    assert _to_tensor(v) is not None
