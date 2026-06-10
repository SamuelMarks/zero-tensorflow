"""TensorFlow nn module."""

from typing import Optional, Any
import numpy as np
import ml_switcheroo.nn as _nn
from . import Tensor, _to_tensor, _wrap

__all__ = ["elu", "leaky_relu", "relu", "selu", "sigmoid", "softmax", "tanh"]


def _check_none(features):
    if features is None:
        raise ValueError("features cannot be None")


def elu(features: Any, *args: Any, name: Optional[str] = None, **kwargs: Any) -> Tensor:
    """Computes the exponential linear function."""
    _check_none(features)
    return _wrap(_nn.elu(_to_tensor(features), *args, **kwargs))


def leaky_relu(
    features: Any,
    *args: Any,
    alpha: float = 0.2,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Tensor:
    """Compute the Leaky ReLU activation function."""
    _check_none(features)
    return _wrap(
        _nn.leaky_relu(_to_tensor(features), negative_slope=alpha, *args, **kwargs)
    )


def relu(
    features: Any, *args: Any, name: Optional[str] = None, **kwargs: Any
) -> Tensor:
    """Computes rectified linear: max(features, 0)."""
    _check_none(features)
    return _wrap(_nn.relu(_to_tensor(features), *args, **kwargs))


def selu(
    features: Any, *args: Any, name: Optional[str] = None, **kwargs: Any
) -> Tensor:
    """Computes scaled exponential linear: scale * alpha * (exp(features) - 1)."""
    _check_none(features)
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(features)
        scale = 1.0507009873554804934193349852946
        alpha_val = 1.6732632423543772848170429916717
        res = scale * np.where(data > 0, data, alpha_val * (np.exp(data) - 1))
        return _wrap(_to_tensor(res))
    return _wrap(_nn.selu(_to_tensor(features), *args, **kwargs))


def sigmoid(x: Any, *args: Any, name: Optional[str] = None, **kwargs: Any) -> Tensor:
    """Computes sigmoid of x element-wise."""
    _check_none(x)
    return _wrap(_nn.sigmoid(_to_tensor(x), *args, **kwargs))


def softmax(
    logits: Any,
    *args: Any,
    axis: Optional[int] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Tensor:
    """Computes softmax activations."""
    _check_none(logits)
    if axis is None:
        axis = -1
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(logits)
        e_x = np.exp(data - np.max(data, axis=axis, keepdims=True))
        res = e_x / e_x.sum(axis=axis, keepdims=True)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softmax(_to_tensor(logits), dim=axis, *args, **kwargs))


def tanh(x: Any, *args: Any, name: Optional[str] = None, **kwargs: Any) -> Tensor:
    """Computes hyperbolic tangent of x element-wise."""
    _check_none(x)
    return _wrap(_nn.tanh(_to_tensor(x), *args, **kwargs))
