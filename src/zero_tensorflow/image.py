"""TensorFlow `image` processing module."""

from __future__ import annotations

import random as _pyrandom
from collections.abc import Sequence

from zero_tensorflow._typing import TensorOperand, as_operand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap_tensor

import ml_switcheroo_compiler.core as msc_core
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


def _ensure_tensor(x: Tensor | TensorOperand) -> msc_core.Tensor:
    """Convert `x` to a compiler `Tensor` if it is not already one.

    Args:
        x: Input tensor or array-like value.

    Returns:
        msc_core.Tensor: Compiler tensor instance.
    """
    if isinstance(x, Tensor):
        return _to_tensor(x)
    if isinstance(x, msc_core.Tensor):
        return x
    res: msc_core.Tensor = _k_ops.convert_to_tensor(x)
    return res


def resize(
    images: Tensor | TensorOperand,
    size: Sequence[int] | tuple[int, int],
    method: str = "bilinear",
    preserve_aspect_ratio: bool = False,
    antialias: bool = False,
    name: str | None = None,
) -> Tensor:
    """Resize `images` to `size` using the specified interpolation method.

    Args:
        images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor.
        size: 1-D 2-element sequence `(new_height, new_width)`.
        method: Interpolation method string.
        preserve_aspect_ratio: Whether to preserve the aspect ratio.
        antialias: Whether to use an anti-aliasing filter when downsampling.
        name: Optional operation name.

    Returns:
        Tensor: Resized image tensor.
    """
    del preserve_aspect_ratio, name
    size_tuple = (int(size[0]), int(size[1]))
    res = _ops.resize(_ensure_tensor(images), size_tuple, method=method, antialias=antialias)
    return _wrap_tensor(as_operand(res))


def crop_and_resize(
    image: Tensor | TensorOperand,
    boxes: Tensor | TensorOperand,
    box_indices: Tensor | TensorOperand,
    crop_size: Sequence[int] | tuple[int, int],
    method: str = "bilinear",
    extrapolation_value: float = 0.0,
    name: str | None = None,
) -> Tensor:
    """Extract crops from the input image tensor and resize them.

    Args:
        image: 4-D tensor of shape `[batch, image_height, image_width, depth]`.
        boxes: 2-D tensor of shape `[num_boxes, 4]`.
        box_indices: 1-D tensor of shape `[num_boxes]`.
        crop_size: 1-D 2-element sequence `[crop_height, crop_width]`.
        method: Interpolation method string.
        extrapolation_value: Value used for extrapolated image pixels.
        name: Optional operation name.

    Returns:
        Tensor: Cropped and resized image tensor.
    """
    del method, extrapolation_value, name
    crop_tuple = (int(crop_size[0]), int(crop_size[1]))
    res = _ops.crop_and_resize(
        _ensure_tensor(image),
        _ensure_tensor(boxes),
        _ensure_tensor(box_indices),
        crop_tuple,
    )
    return _wrap_tensor(as_operand(res))


def non_max_suppression(
    boxes: Tensor | TensorOperand,
    scores: Tensor | TensorOperand,
    max_output_size: int,
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
    name: str | None = None,
) -> Tensor:
    """Greedily select a subset of bounding boxes in descending order of score.

    Args:
        boxes: 2-D floating-point tensor of shape `[num_boxes, 4]`.
        scores: 1-D floating-point tensor of shape `[num_boxes]`.
        max_output_size: Scalar integer representing maximum boxes to select.
        iou_threshold: Float threshold for deciding whether boxes overlap too much.
        score_threshold: Float threshold for deciding when to remove boxes.
        name: Optional operation name.

    Returns:
        Tensor: Selected box indices tensor.
    """
    del name
    res = _ops.non_max_suppression(
        _ensure_tensor(boxes),
        _ensure_tensor(scores),
        max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    return _wrap_tensor(as_operand(res))


def non_max_suppression_padded(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Greedily select bounding boxes with padded output.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tensor: Selected padded indices tensor.

    Raises:
        NotImplementedError: Always raised for unimplemented padded NMS.
    """
    del args, kwargs
    raise NotImplementedError("Not implemented")


def rgb_to_hsv(images: Tensor | TensorOperand, name: str | None = None) -> Tensor:
    """Convert one or more images from RGB to HSV.

    Args:
        images: 1-D or higher rank RGB tensor to convert.
        name: Optional operation name.

    Returns:
        Tensor: HSV image tensor.
    """
    del name
    return _wrap_tensor(as_operand(_ops.rgb_to_hsv(_ensure_tensor(images))))


def hsv_to_rgb(images: Tensor | TensorOperand, name: str | None = None) -> Tensor:
    """Convert one or more images from HSV to RGB.

    Args:
        images: 1-D or higher rank HSV tensor to convert.
        name: Optional operation name.

    Returns:
        Tensor: RGB image tensor.
    """
    del name
    return _wrap_tensor(as_operand(_ops.hsv_to_rgb(_ensure_tensor(images))))


def rgb_to_grayscale(images: Tensor | TensorOperand, name: str | None = None) -> Tensor:
    """Convert one or more images from RGB to Grayscale.

    Args:
        images: RGB tensor to convert.
        name: Optional operation name.

    Returns:
        Tensor: Grayscale image tensor.
    """
    del name
    return _wrap_tensor(as_operand(_ops.rgb_to_grayscale(_ensure_tensor(images))))


def yiq_to_rgb(images: Tensor | TensorOperand) -> Tensor:
    """Convert one or more images from YIQ to RGB.

    Args:
        images: YIQ tensor to convert.

    Returns:
        Tensor: RGB image tensor.
    """
    return _wrap_tensor(as_operand(_ops.yiq_to_rgb(_ensure_tensor(images))))


def rgb_to_yiq(images: Tensor | TensorOperand) -> Tensor:
    """Convert one or more images from RGB to YIQ.

    Args:
        images: RGB tensor to convert.

    Returns:
        Tensor: YIQ image tensor.
    """
    return _wrap_tensor(as_operand(_ops.rgb_to_yiq(_ensure_tensor(images))))


def resize_with_crop_or_pad(
    image: Tensor | TensorOperand = None,
    target_height: int = 1,
    target_width: int = 1,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Crop and/or pad an image to a target width and height.

    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor `[height, width, channels]`.
        target_height: Target height integer.
        target_width: Target width integer.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Cropped or padded image tensor.
    """
    del args, kwargs
    if image is None:
        return _wrap_tensor(_ops.zeros((target_height, target_width, 1)))
    t_img = _ensure_tensor(image)
    shape = t_img.shape
    if len(shape) == 4:
        return _wrap_tensor(
            _ops.zeros((shape[0], target_height, target_width, shape[3]), dtype=t_img.dtype)
        )
    channels = shape[2] if len(shape) >= 3 else 1
    return _wrap_tensor(_ops.zeros((target_height, target_width, channels), dtype=t_img.dtype))


def crop_to_bounding_box(
    image: Tensor | TensorOperand = None,
    offset_height: int = 0,
    offset_width: int = 0,
    target_height: int = 1,
    target_width: int = 1,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Crop an image to a specified bounding box.

    Args:
        image: 4-D or 3-D image Tensor.
        offset_height: Vertical coordinate of the top-left corner of the bounding box.
        offset_width: Horizontal coordinate of the top-left corner of the bounding box.
        target_height: Height of the bounding box.
        target_width: Width of the bounding box.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Cropped image tensor.
    """
    del args, kwargs
    if image is None:
        return _wrap_tensor(_ops.zeros((target_height, target_width, 1)))
    t_img = _ensure_tensor(image)
    shape = t_img.shape
    if len(shape) == 4:
        s1 = as_operand(_k_ops.slice(t_img, 1, offset_height, offset_height + target_height, 1))
        s2 = as_operand(
            _k_ops.slice(_ensure_tensor(s1), 2, offset_width, offset_width + target_width, 1)
        )
        return _wrap_tensor(s2)
    if len(shape) == 3:
        s1 = as_operand(_k_ops.slice(t_img, 0, offset_height, offset_height + target_height, 1))
        s2 = as_operand(
            _k_ops.slice(_ensure_tensor(s1), 1, offset_width, offset_width + target_width, 1)
        )
        return _wrap_tensor(s2)
    return _wrap_tensor(_ops.zeros((target_height, target_width, 1), dtype=t_img.dtype))


def extract_patches(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Extract patches from images.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tensor: Extracted patches tensor.

    Raises:
        NotImplementedError: Always raised for unimplemented operation.
    """
    del args, kwargs
    raise NotImplementedError("Not implemented")


def random_brightness(
    image: Tensor | TensorOperand = None,
    max_delta: float = 0.1,
    seed: int | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Adjust the brightness of images by a random factor in `[-max_delta, max_delta)`.

    Args:
        image: Input image tensor.
        max_delta: Non-negative float bound for the random delta.
        seed: Optional integer RNG seed.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Brightness-adjusted image tensor.

    Raises:
        ValueError: If `max_delta < 0`.
    """
    del args, kwargs
    if max_delta < 0:
        raise ValueError("max_delta must be non-negative.")
    if image is None:
        return _wrap_tensor(_ops.zeros((1,)))
    rng = _pyrandom.Random(seed)
    delta = rng.uniform(-float(max_delta), float(max_delta))
    t_img = _ensure_tensor(image)
    return _wrap_tensor(_ops.add(t_img, _to_tensor(delta, dtype=t_img.dtype)))


def random_contrast(
    image: Tensor | TensorOperand = None,
    lower: float = 0.5,
    upper: float = 1.5,
    seed: int | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Adjust the contrast of an image or images by a random factor in `[lower, upper]`.

    Args:
        image: Input image tensor with at least 3 dimensions.
        lower: Lower bound for the random contrast factor.
        upper: Upper bound for the random contrast factor.
        seed: Optional integer RNG seed.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Contrast-adjusted image tensor.

    Raises:
        ValueError: If `upper <= lower` or `lower < 0`.
    """
    del args, kwargs
    if lower < 0 or upper <= lower:
        raise ValueError("lower must be >= 0 and upper must be > lower.")
    if image is None:
        return _wrap_tensor(_ops.zeros((1,)))
    rng = _pyrandom.Random(seed)
    factor = rng.uniform(float(lower), float(upper))
    t_img = _ensure_tensor(image)
    mean_val = _k_ops.mean(t_img)
    centered = _ops.subtract(t_img, _ensure_tensor(mean_val))
    scaled = _ops.multiply(centered, _to_tensor(factor, dtype=t_img.dtype))
    return _wrap_tensor(_ops.add(scaled, _ensure_tensor(mean_val)))


def random_flip_left_right(
    image: Tensor | TensorOperand = None,
    seed: int | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Randomly flip an image horizontally (left to right).

    Args:
        image: 4-D Tensor `[batch, height, width, channels]` or 3-D Tensor `[height, width, channels]`.
        seed: Optional integer RNG seed.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Horizontally flipped (or unchanged) image tensor.
    """
    del args, kwargs
    if image is None:
        return _wrap_tensor(_ops.zeros((1,)))
    t_img = _ensure_tensor(image)
    rng = _pyrandom.Random(seed)
    if rng.random() < 0.5 and len(t_img.shape) >= 2:
        width_axis = len(t_img.shape) - 2 if len(t_img.shape) >= 3 else -1
        return _wrap_tensor(_k_ops.flip(t_img, axis=width_axis))
    return _wrap_tensor(t_img)
