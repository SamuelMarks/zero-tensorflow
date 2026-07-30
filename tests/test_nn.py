import numpy as np
import pytest

from zero_tensorflow import nn


def test_elu():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.elu(x)
    np.testing.assert_allclose(
        y.numpy(), np.array([np.exp(-1.0) - 1.0, 0.0, 1.0], dtype=np.float32), rtol=1e-5
    )

    with pytest.raises(ValueError):
        nn.elu(None)  # type: ignore


def test_leaky_relu():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.leaky_relu(x, alpha=0.1)
    np.testing.assert_allclose(
        y.numpy(), np.array([-0.1, 0.0, 1.0], dtype=np.float32), rtol=1e-5
    )

    with pytest.raises(ValueError):
        nn.leaky_relu(None)  # type: ignore


def test_relu():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.relu(x)
    np.testing.assert_allclose(
        y.numpy(), np.array([0.0, 0.0, 1.0], dtype=np.float32), rtol=1e-5
    )

    with pytest.raises(ValueError):
        nn.relu(None)  # type: ignore


def test_selu():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.selu(x)
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    expected_neg = scale * alpha * (np.exp(-1.0) - 1.0)
    expected = np.array([expected_neg, 0.0, scale * 1.0], dtype=np.float32)
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)

    with pytest.raises(ValueError):
        nn.selu(None)  # type: ignore


def test_sigmoid():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.sigmoid(x)
    np.testing.assert_allclose(y.numpy(), 1 / (1 + np.exp(-x)), rtol=1e-5)

    with pytest.raises(ValueError):
        nn.sigmoid(None)  # type: ignore


def test_softmax():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = nn.softmax(x)
    e_x = np.exp(x - 3.0)
    expected = e_x / np.sum(e_x, axis=-1, keepdims=True)
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)

    # test default axis
    y_default = nn.softmax(x, axis=None)
    np.testing.assert_allclose(y_default.numpy(), expected, rtol=1e-5)

    with pytest.raises(ValueError):
        nn.softmax(None)  # type: ignore


def test_tanh():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.tanh(x)
    np.testing.assert_allclose(y.numpy(), np.tanh(x), rtol=1e-5)

    with pytest.raises(ValueError):
        nn.tanh(None)  # type: ignore


def test_nn_traced():
    from zero_tensorflow import Tensor, function

    @function
    def nn_f(x):
        return (
            nn.relu(x),
            nn.elu(x),
            nn.leaky_relu(x),
            nn.selu(x),
            nn.sigmoid(x),
            nn.softmax(x),
            nn.tanh(x),
        )

    x = Tensor([-1.0, 0.0, 1.0])
    res = nn_f(x)
    assert all(isinstance(r, Tensor) for r in res)
