"""TensorFlow nn module."""

from __future__ import annotations

from typing import Any

from zero_keras import activations as k_activations
from zero_keras import ops as k_ops

from . import Tensor, _to_tensor, _wrap

__all__ = [
    "avg_pool3d",
    "batch_norm_with_global_normalization",
    "conv1d",
    "conv2d_transpose",
    "conv3d_transpose",
    "crelu",
    "ctc_loss",
    "depthwise_conv2d",
    "elu",
    "fractional_max_pool",
    "gelu",
    "leaky_relu",
    "local_response_normalization",
    "max_pool3d",
    "nce_loss",
    "relu",
    "sampled_softmax_loss",
    "selu",
    "separable_conv2d",
    "sigmoid",
    "silu",
    "softmax",
    "swish",
    "tanh",
]


def _check_none(features):
    """
    Check if features are None and raise an error if so.
    """
    if features is None:
        raise ValueError("features cannot be None")


def _unwrap_k(res):
    if hasattr(res, "data"):
        return res.data
    return res


def avg_pool3d(*args, **kwargs) -> Tensor:
    """Computes 3D average pooling."""
    return _wrap(_unwrap_k(k_ops.avg_pool3d(*args, **kwargs)))


def batch_norm_with_global_normalization(*args, **kwargs) -> Tensor:
    """Batch normalization with global normalization."""
    raise NotImplementedError(
        "batch_norm_with_global_normalization is not yet implemented in ml-switcheroo-compiler"
    )


def conv1d(*args, **kwargs) -> Tensor:
    """Computes a 1-D convolution."""
    return _wrap(_unwrap_k(k_ops.conv1d(*args, **kwargs)))


def conv2d_transpose(*args, **kwargs) -> Tensor:
    """Computes a 2-D transposed convolution."""
    raise NotImplementedError(
        "conv2d_transpose is not yet implemented in ml-switcheroo-compiler"
    )


def conv3d_transpose(*args, **kwargs) -> Tensor:
    """Computes a 3-D transposed convolution."""
    raise NotImplementedError(
        "conv3d_transpose is not yet implemented in ml-switcheroo-compiler"
    )


def crelu(features: Any, *args: Any, name: str | None = None, **kwargs: Any) -> Tensor:
    """Computes Concatenated ReLU."""
    _check_none(features)
    t = _to_tensor(features)
    from . import _ops

    pos = _ops.maximum(t, _to_tensor(0.0, dtype=t.dtype))
    neg = _ops.maximum(_ops.negative(t), _to_tensor(0.0, dtype=t.dtype))
    return _wrap(_ops.concat([pos, neg], axis=-1))


def ctc_loss(*args, **kwargs) -> Tensor:
    """Computes CTC (Connectionist Temporal Classification) loss."""
    return _wrap(_unwrap_k(k_ops.ctc_loss(*args, **kwargs)))


def depthwise_conv2d(*args, **kwargs) -> Tensor:
    """Computes a depthwise 2-D convolution."""
    return _wrap(_unwrap_k(k_ops.depthwise_conv(*args, **kwargs)))


def elu(features: Any, *args: Any, name: str | None = None, **kwargs: Any) -> Tensor:
    """Computes the exponential linear function."""
    _check_none(features)
    t = _to_tensor(features)
    alpha_t = _to_tensor(1.0, dtype=t.dtype)
    from . import _ops

    cond = _ops.greater(t, _to_tensor(0.0, dtype=t.dtype))
    cond_f = _ops.cast(cond, t.dtype)
    not_cond_f = _ops.subtract(_to_tensor(1.0, dtype=t.dtype), cond_f)

    res = _ops.add(
        _ops.multiply(cond_f, t),
        _ops.multiply(
            not_cond_f,
            _ops.multiply(
                alpha_t, _ops.subtract(_ops.exp(t), _to_tensor(1.0, dtype=t.dtype))
            ),
        ),
    )
    return _wrap(res)


def fractional_max_pool(*args, **kwargs) -> Tensor:
    """Performs fractional max pooling."""
    raise NotImplementedError(
        "fractional_max_pool is not yet implemented in ml-switcheroo-compiler"
    )


def gelu(
    features: Any, approximate: bool = False, name: str | None = None, **kwargs: Any
) -> Tensor:
    """Computes the Gaussian Error Linear Unit (GELU) activation function."""
    _check_none(features)
    return _wrap(
        _unwrap_k(
            k_activations.gelu(_to_tensor(features), approximate=approximate, **kwargs)
        )
    )


def leaky_relu(
    features: Any, alpha: float = 0.2, name: str | None = None, **kwargs: Any
) -> Tensor:
    """Compute the Leaky ReLU activation function."""
    _check_none(features)
    return _wrap(
        _unwrap_k(
            k_activations.leaky_relu(
                _to_tensor(features), negative_slope=alpha, **kwargs
            )
        )
    )


def local_response_normalization(*args, **kwargs) -> Tensor:
    """Local Response Normalization."""
    raise NotImplementedError(
        "local_response_normalization is not yet implemented in ml-switcheroo-compiler"
    )


def max_pool3d(*args, **kwargs) -> Tensor:
    """Computes 3D max pooling."""
    return _wrap(_unwrap_k(k_ops.max_pool3d(*args, **kwargs)))


def nce_loss(*args, **kwargs) -> Tensor:
    """Computes NCE loss."""
    raise NotImplementedError(
        "nce_loss is not yet implemented in ml-switcheroo-compiler"
    )


def relu(features: Any, *args: Any, name: str | None = None, **kwargs: Any) -> Tensor:
    """Computes rectified linear: max(features, 0)."""
    _check_none(features)
    return _wrap(_unwrap_k(k_activations.relu(_to_tensor(features), *args, **kwargs)))


def sampled_softmax_loss(*args, **kwargs) -> Tensor:
    """Computes and returns the sampled softmax training loss."""
    raise NotImplementedError(
        "sampled_softmax_loss is not yet implemented in ml-switcheroo-compiler"
    )


def selu(features: Any, *args: Any, name: str | None = None, **kwargs: Any) -> Tensor:
    """Computes scaled exponential linear: scale * alpha * (exp(features) - 1)."""
    _check_none(features)
    t = _to_tensor(features)
    from . import _ops

    scale = _to_tensor(1.0507009873554804934193349852946, dtype=t.dtype)
    alpha = _to_tensor(1.6732632423543772848170429916717, dtype=t.dtype)
    cond = _ops.greater(t, _to_tensor(0.0, dtype=t.dtype))
    cond_f = _ops.cast(cond, t.dtype)
    not_cond_f = _ops.subtract(_to_tensor(1.0, dtype=t.dtype), cond_f)

    res = _ops.add(
        _ops.multiply(cond_f, t),
        _ops.multiply(
            not_cond_f,
            _ops.multiply(
                alpha, _ops.subtract(_ops.exp(t), _to_tensor(1.0, dtype=t.dtype))
            ),
        ),
    )
    return _wrap(_ops.multiply(scale, res))


def separable_conv2d(*args, **kwargs) -> Tensor:
    """Computes a 2-D separable convolution."""
    return _wrap(_unwrap_k(k_ops.separable_conv(*args, **kwargs)))


def sigmoid(x: Any, *args: Any, name: str | None = None, **kwargs: Any) -> Tensor:
    """Computes sigmoid of x element-wise."""
    _check_none(x)
    return _wrap(_unwrap_k(k_activations.sigmoid(_to_tensor(x), *args, **kwargs)))


def silu(features: Any, *args: Any, name: str | None = None, **kwargs: Any) -> Tensor:
    """Computes the SiLU activation function."""
    _check_none(features)
    return _wrap(_unwrap_k(k_activations.silu(_to_tensor(features), *args, **kwargs)))


def softmax(
    logits: Any, axis: int | None = None, name: str | None = None, **kwargs: Any
) -> Tensor:
    """Computes softmax activations."""
    _check_none(logits)
    if axis is None:
        axis = -1
    return _wrap(
        _unwrap_k(k_activations.softmax(_to_tensor(logits), axis=axis, **kwargs))
    )


def swish(features: Any, *args: Any, name: str | None = None, **kwargs: Any) -> Tensor:
    """Computes the swish activation function."""
    _check_none(features)
    return _wrap(_unwrap_k(k_activations.swish(_to_tensor(features), *args, **kwargs)))


def tanh(x: Any, *args: Any, name: str | None = None, **kwargs: Any) -> Tensor:
    """Computes hyperbolic tangent of x element-wise."""
    _check_none(x)
    return _wrap(_unwrap_k(k_activations.tanh(_to_tensor(x), *args, **kwargs)))
