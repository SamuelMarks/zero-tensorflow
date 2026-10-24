"""Unit tests for `test_nn`."""

from __future__ import annotations

from zero_tensorflow import Tensor, function, nn

import numpy as np
import pytest


def test_elu() -> None:
    """Execute `test_elu`."""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.elu(x).numpy()
    np.testing.assert_allclose(
        y, np.array([np.exp(-1.0) - 1.0, 0.0, 1.0], dtype=np.float32), rtol=1e-05
    )
    with pytest.raises(ValueError):
        nn.elu(None)


def test_leaky_relu() -> None:
    """Execute `test_leaky_relu`."""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.leaky_relu(x, alpha=0.1).numpy()
    np.testing.assert_allclose(y, np.array([-0.1, 0.0, 1.0], dtype=np.float32), rtol=1e-05)
    with pytest.raises(ValueError):
        nn.leaky_relu(None)


def test_relu() -> None:
    """Execute `test_relu`."""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.relu(x).numpy()
    np.testing.assert_allclose(y, np.array([0.0, 0.0, 1.0], dtype=np.float32), rtol=1e-05)
    with pytest.raises(ValueError):
        nn.relu(None)


def test_selu() -> None:
    """Execute `test_selu`."""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.selu(x).numpy()
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    expected_neg = scale * alpha * (np.exp(-1.0) - 1.0)
    expected = np.array([expected_neg, 0.0, scale * 1.0], dtype=np.float32)
    np.testing.assert_allclose(y, expected, rtol=1e-05)
    with pytest.raises(ValueError):
        nn.selu(None)


def test_sigmoid() -> None:
    """Execute `test_sigmoid`."""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.sigmoid(x).numpy()
    np.testing.assert_allclose(y, 1 / (1 + np.exp(-x)), rtol=1e-05)
    with pytest.raises(ValueError):
        nn.sigmoid(None)


def test_softmax() -> None:
    """Execute `test_softmax`."""
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = nn.softmax(x).numpy()
    e_x = np.exp(x - 3.0)
    expected = e_x / np.sum(e_x, axis=-1, keepdims=True)
    np.testing.assert_allclose(y, expected, rtol=1e-05)
    y_default = nn.softmax(x, axis=None).numpy()
    np.testing.assert_allclose(y_default, expected, rtol=1e-05)
    with pytest.raises(ValueError):
        nn.softmax(None)


def test_tanh() -> None:
    """Execute `test_tanh`."""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = nn.tanh(x).numpy()
    np.testing.assert_allclose(y, np.tanh(x), rtol=1e-05)
    with pytest.raises(ValueError):
        nn.tanh(None)


def test_nn_traced() -> None:
    """Execute `test_nn_traced`."""

    @function
    def nn_f(x: Tensor) -> tuple[Tensor, ...]:
        """Execute `nn_f`.

        Args:
            x: Parameter `x`.

        Returns:
            tuple[Tensor, ...]: Computed result.
        """
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


def test_nn_conv_pool_norm_and_losses() -> None:
    """Verify `nn` convolutions, pooling, batch/LRN normalization, and loss functions."""
    x3d = Tensor(np.ones((1, 2, 2, 2, 1), dtype=np.float32))
    assert isinstance(nn.avg_pool3d(x3d, (1, 1, 1), (1, 1, 1), "VALID"), Tensor)
    assert isinstance(nn.max_pool3d(x3d, (1, 1, 1), (1, 1, 1), "VALID"), Tensor)

    x1d = Tensor(np.ones((1, 2, 4), dtype=np.float32))
    k1d = Tensor(np.ones((3, 2, 2), dtype=np.float32))
    assert isinstance(nn.conv1d(x1d, k1d), Tensor)

    x2d = Tensor(np.ones((1, 2, 2, 2), dtype=np.float32))
    k2d = Tensor(np.ones((2, 2, 2, 2), dtype=np.float32))
    assert isinstance(nn.conv2d_transpose(x2d, k2d, 1, "VALID"), Tensor)
    with pytest.raises(NotImplementedError):
        nn.conv2d_transpose()

    x3d_c = Tensor(np.ones((1, 1, 2, 2, 2), dtype=np.float32))
    k3d = Tensor(np.ones((1, 1, 1, 1, 1), dtype=np.float32))
    assert isinstance(nn.conv3d_transpose(x3d_c, k3d, 1, "VALID"), Tensor)
    with pytest.raises(NotImplementedError):
        nn.conv3d_transpose()

    x2d_hwc = Tensor(np.ones((1, 4, 4, 2), dtype=np.float32))
    dw_k = Tensor(np.ones((2, 2, 2, 1), dtype=np.float32))
    assert isinstance(nn.depthwise_conv2d(x2d_hwc, dw_k), Tensor)
    pt_k = Tensor(np.ones((1, 1, 2, 3), dtype=np.float32))
    assert isinstance(nn.separable_conv2d(x2d_hwc, dw_k, pt_k), Tensor)

    crelu_out = nn.crelu(Tensor([-1.0, 2.0]))
    np.testing.assert_allclose(crelu_out.numpy(), [0.0, 2.0, 1.0, 0.0])

    assert isinstance(nn.gelu(Tensor([0.0, 1.0])), Tensor)
    assert isinstance(nn.silu(Tensor([0.0, 1.0])), Tensor)
    assert isinstance(nn.swish(Tensor([0.0, 1.0])), Tensor)

    bn_out = nn.batch_norm_with_global_normalization(
        Tensor([2.0, 4.0]),
        Tensor([1.0, 1.0]),
        Tensor([1.0, 1.0]),
        Tensor([0.0, 0.0]),
        Tensor([2.0, 2.0]),
        variance_epsilon=0.0,
        scale_after_normalization=True,
    )
    np.testing.assert_allclose(bn_out.numpy(), [2.0, 6.0])

    bn_unscaled = nn.batch_norm_with_global_normalization(
        Tensor([2.0, 4.0]),
        Tensor([1.0, 1.0]),
        Tensor([1.0, 1.0]),
        Tensor([0.0, 0.0]),
        variance_epsilon=0.0,
        scale_after_normalization=False,
    )
    np.testing.assert_allclose(bn_unscaled.numpy(), [1.0, 3.0])
    with pytest.raises(NotImplementedError):
        nn.batch_norm_with_global_normalization()

    lrn_out = nn.local_response_normalization(Tensor([0.0, 1.0]), bias=1.0, alpha=1.0, beta=1.0)
    np.testing.assert_allclose(lrn_out.numpy(), [0.0, 0.5])
    with pytest.raises(NotImplementedError):
        nn.local_response_normalization()

    y_true_ctc = Tensor([[1, 2]])
    y_pred_ctc = Tensor(np.ones((4, 1, 4), dtype=np.float32))
    assert isinstance(
        nn.ctc_loss(y_true_ctc, y_pred_ctc, Tensor([2]), Tensor([4])),
        Tensor,
    )

    with pytest.raises(NotImplementedError):
        nn.fractional_max_pool()
    with pytest.raises(NotImplementedError):
        nn.nce_loss()
    with pytest.raises(NotImplementedError):
        nn.sampled_softmax_loss()
