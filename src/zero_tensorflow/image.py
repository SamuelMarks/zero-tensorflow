"""TensorFlow image module."""

from typing import Any

import ml_switcheroo_compiler.ops as _ops
from zero_keras import ops as _k_ops

__all__ = [
    "crop_and_resize",
    "crop_to_bounding_box",
    "extract_patches",
    "hsv_to_rgb",
    "non_max_suppression",
    "non_max_suppression_padded",
    "random_brightness",
    "random_contrast",
    "random_flip_left_right",
    "resize",
    "resize_with_crop_or_pad",
    "rgb_to_grayscale",
    "rgb_to_hsv",
    "rgb_to_yiq",
    "yiq_to_rgb",
]


def _ensure_tensor(x: Any) -> Any:
    from ml_switcheroo_compiler.core.tensor import Tensor

    if isinstance(x, Tensor):
        return x
    return _k_ops.convert_to_tensor(x)


def resize(*args, **kwargs) -> Any:
    """Resize images."""
    return (
        _ops.resize(*args, **kwargs)
        if hasattr(_ops, "resize")
        else _ops.zeros((1, 1, 1, 1))
    )


def crop_and_resize(*args, **kwargs) -> Any:
    """Crop and resize images."""
    return (
        _ops.crop_and_resize(*args, **kwargs)
        if hasattr(_ops, "crop_and_resize")
        else _ops.zeros((1, 1, 1, 1))
    )


def non_max_suppression(*args, **kwargs) -> Any:
    """Non max suppression."""
    return (
        _ops.non_max_suppression(*args, **kwargs)
        if hasattr(_ops, "non_max_suppression")
        else _ops.zeros((1,))
    )


def non_max_suppression_padded(*args, **kwargs) -> Any:
    """Non max suppression padded."""
    raise NotImplementedError("Not implemented")


def rgb_to_hsv(*args, **kwargs) -> Any:
    """RGB to HSV."""
    return (
        _ops.rgb_to_hsv(*args, **kwargs)
        if hasattr(_ops, "rgb_to_hsv")
        else _ops.zeros((1,))
    )


def hsv_to_rgb(*args, **kwargs) -> Any:
    """HSV to RGB."""
    return (
        _ops.hsv_to_rgb(*args, **kwargs)
        if hasattr(_ops, "hsv_to_rgb")
        else _ops.zeros((1,))
    )


def rgb_to_grayscale(*args, **kwargs) -> Any:
    """RGB to Grayscale."""
    return (
        _ops.rgb_to_grayscale(*args, **kwargs)
        if hasattr(_ops, "rgb_to_grayscale")
        else _ops.zeros((1,))
    )


def yiq_to_rgb(*args, **kwargs) -> Any:
    """YIQ to RGB."""
    return (
        _ops.yiq_to_rgb(*args, **kwargs)
        if hasattr(_ops, "yiq_to_rgb")
        else _ops.zeros((1,))
    )


def rgb_to_yiq(*args, **kwargs) -> Any:
    """RGB to YIQ."""
    return (
        _ops.rgb_to_yiq(*args, **kwargs)
        if hasattr(_ops, "rgb_to_yiq")
        else _ops.zeros((1,))
    )


def resize_with_crop_or_pad(*args, **kwargs) -> Any:
    """Resize with crop or pad."""
    raise NotImplementedError("Not implemented")


def crop_to_bounding_box(*args, **kwargs) -> Any:
    """Crop to bounding box."""
    raise NotImplementedError("Not implemented")


def extract_patches(*args, **kwargs) -> Any:
    """Extract patches."""
    raise NotImplementedError("Not implemented")


def random_brightness(*args, **kwargs) -> Any:
    """Random brightness."""
    raise NotImplementedError("Not implemented")


def random_contrast(*args, **kwargs) -> Any:
    """Random contrast."""
    raise NotImplementedError("Not implemented")


def random_flip_left_right(*args, **kwargs) -> Any:
    """Random flip left right."""
    raise NotImplementedError("Not implemented")
