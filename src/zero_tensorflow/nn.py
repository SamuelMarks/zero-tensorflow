"""TensorFlow nn module."""

from typing import Optional, Any
import ml_switcheroo.nn as _nn
from . import Tensor, _to_tensor, _wrap

__all__ = ["elu", "leaky_relu", "relu", "selu", "sigmoid", "softmax", "tanh"]


def _check_none(features):
    """
    Check if features are None and raise an error if so.

    Args:
        features (Any): The features to check.

    Raises:
        ValueError: If features is None.
    """
    if features is None:
        raise ValueError("features cannot be None")


def elu(features: Any, *args: Any, name: Optional[str] = None, **kwargs: Any) -> Tensor:
    """
    Computes the exponential linear function.

    Args:
        features (Any): The input features.
        *args (Any): Variable length argument list.
        name (Optional[str]): Optional name for the operation.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Tensor: A Tensor representing the exponential linear function of the input.
    """
    _check_none(features)
    return _wrap(_nn.elu(_to_tensor(features), *args, **kwargs))


def leaky_relu(
    features: Any,
    *args: Any,
    alpha: float = 0.2,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Tensor:
    """
    Compute the Leaky ReLU activation function.

    Args:
        features (Any): The input features.
        *args (Any): Variable length argument list.
        alpha (float): Slope of the activation function at x < 0.
        name (Optional[str]): Optional name for the operation.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Tensor: A Tensor representing the Leaky ReLU activation of the input.
    """
    _check_none(features)
    return _wrap(
        _nn.leaky_relu(_to_tensor(features), negative_slope=alpha, *args, **kwargs)
    )


def relu(
    features: Any, *args: Any, name: Optional[str] = None, **kwargs: Any
) -> Tensor:
    """
    Computes rectified linear: max(features, 0).

    Args:
        features (Any): The input features.
        *args (Any): Variable length argument list.
        name (Optional[str]): Optional name for the operation.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Tensor: A Tensor representing the rectified linear activation of the input.
    """
    _check_none(features)
    return _wrap(_nn.relu(_to_tensor(features), *args, **kwargs))


def selu(
    features: Any, *args: Any, name: Optional[str] = None, **kwargs: Any
) -> Tensor:
    """
    Computes scaled exponential linear: scale * alpha * (exp(features) - 1).

    Args:
        features (Any): The input features.
        *args (Any): Variable length argument list.
        name (Optional[str]): Optional name for the operation.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Tensor: A Tensor representing the scaled exponential linear activation of the input.
    """
    _check_none(features)
    t = _to_tensor(features)
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        import numpy as np
        from ml_switcheroo.core.tensor_utils import to_array

        x = to_array(t)
        scale = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717
        res = scale * np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))
        return _wrap(_to_tensor(res))
    return _wrap(_nn.selu(t, *args, **kwargs))


def sigmoid(x: Any, *args: Any, name: Optional[str] = None, **kwargs: Any) -> Tensor:
    """
    Computes sigmoid of x element-wise.

    Args:
        x (Any): The input tensor.
        *args (Any): Variable length argument list.
        name (Optional[str]): Optional name for the operation.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Tensor: A Tensor representing the sigmoid activation of the input.
    """
    _check_none(x)
    return _wrap(_nn.sigmoid(_to_tensor(x), *args, **kwargs))


def softmax(
    logits: Any,
    *args: Any,
    axis: Optional[int] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Tensor:
    """
    Computes softmax activations.

    Args:
        logits (Any): The input logits.
        *args (Any): Variable length argument list.
        axis (Optional[int]): The dimension softmax would be performed on.
        name (Optional[str]): Optional name for the operation.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Tensor: A Tensor representing the softmax activation of the input.
    """
    _check_none(logits)
    if axis is None:
        axis = -1
    t = _to_tensor(logits)
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        import numpy as np
        from ml_switcheroo.core.tensor_utils import to_array

        x = to_array(t)
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        res = e_x / e_x.sum(axis=axis, keepdims=True)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softmax(t, dim=axis, *args, **kwargs))


def tanh(x: Any, *args: Any, name: Optional[str] = None, **kwargs: Any) -> Tensor:
    """
    Computes hyperbolic tangent of x element-wise.

    Args:
        x (Any): The input tensor.
        *args (Any): Variable length argument list.
        name (Optional[str]): Optional name for the operation.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        Tensor: A Tensor representing the hyperbolic tangent activation of the input.
    """
    _check_none(x)
    return _wrap(_nn.tanh(_to_tensor(x), *args, **kwargs))
