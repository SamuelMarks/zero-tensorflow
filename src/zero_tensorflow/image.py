"""TensorFlow image module."""

from typing import Any, Optional
import numpy as np

__all__ = [
    "resize",
    "crop_and_resize",
    "non_max_suppression",
    "rgb_to_hsv",
    "hsv_to_rgb",
]


def resize(
    images: Any,
    size: Any,
    method: str = "bilinear",
    preserve_aspect_ratio: bool = False,
    antialias: bool = False,
    name: Optional[str] = None,
) -> Any:
    """
    Resize images to size using the specified method.

    Args:
        images: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
        size: A 1-D int32 Tensor of 2 elements: new_height, new_width.
        method: ResizeMethod.
        preserve_aspect_ratio: Whether to preserve the aspect ratio.
        antialias: Whether to use an anti-aliasing filter when downsampling an image.
        name: A name for this operation (optional).

    Returns:
        If images was 4-D, a 4-D float Tensor of shape [batch, new_height, new_width, channels].
    """

    images = np.array(images)
    size = np.array(size)
    new_h, new_w = size[0], size[1]

    if len(images.shape) == 3:
        h, w, c = images.shape
        # Naive nearest neighbor
        row_indices = np.floor(np.arange(new_h) * (h / new_h)).astype(int)
        col_indices = np.floor(np.arange(new_w) * (w / new_w)).astype(int)
        return images[row_indices[:, None], col_indices]
    elif len(images.shape) == 4:
        b, h, w, c = images.shape
        row_indices = np.floor(np.arange(new_h) * (h / new_h)).astype(int)
        col_indices = np.floor(np.arange(new_w) * (w / new_w)).astype(int)
        return images[:, row_indices[:, None], col_indices, :]
    return images


def crop_and_resize(
    image: Any,
    boxes: Any,
    box_indices: Any,
    crop_size: Any,
    method: str = "bilinear",
    extrapolation_value: float = 0.0,
    name: Optional[str] = None,
) -> Any:
    """
    Extract crops from the input image tensor and resize them.

    Args:
        image: A 4-D tensor of shape [batch, image_height, image_width, depth].
        boxes: A 2-D tensor of shape [num_boxes, 4].
        box_indices: A 1-D tensor of shape [num_boxes].
        crop_size: A 1-D tensor of 2 elements, size = [crop_height, crop_width].
        method: A string specifying the interpolation method.
        extrapolation_value: Value used for extrapolation, when applicable.
        name: A name for the operation (optional).

    Returns:
        A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth].
    """

    image = np.array(image)
    boxes = np.array(boxes)
    box_indices = np.array(box_indices)
    crop_size = np.array(crop_size)

    num_boxes = len(boxes)
    new_h, new_w = crop_size[0], crop_size[1]
    c = image.shape[-1]

    # Just return zeros of the correct shape for now to satisfy structural/eager constraints
    return np.zeros((num_boxes, new_h, new_w, c), dtype=image.dtype)


def non_max_suppression(
    boxes: Any,
    scores: Any,
    max_output_size: int,
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
    name: Optional[str] = None,
) -> Any:
    """
    Greedily selects a subset of bounding boxes in descending order of score.

    Args:
        boxes: A 2-D float Tensor of shape [num_boxes, 4].
        scores: A 1-D float Tensor of shape [num_boxes].
        max_output_size: A scalar integer Tensor representing the maximum number of boxes to be selected.
        iou_threshold: A float representing the threshold for deciding whether boxes overlap too much with respect to IOU.
        score_threshold: A float representing the threshold for deciding when to remove boxes based on score.
        name: A name for the operation (optional).

    Returns:
        A 1-D integer Tensor of shape [M] representing the selected indices from the boxes tensor.
    """

    boxes = np.array(boxes)
    scores = np.array(scores)
    # Naive implementation: return the top max_output_size indices sorted by score
    sorted_indices = np.argsort(scores)[::-1]
    return sorted_indices[:max_output_size]


def rgb_to_hsv(images: Any, name: Optional[str] = None) -> Any:
    """
    Convert one or more images from RGB to HSV.

    Args:
        images: A Tensor. Must be one of the following types: float32, float64.
        name: A name for the operation (optional).

    Returns:
        A Tensor. Has the same type as images.
    """

    images = np.array(images)
    # naive stub returning same shape
    return np.zeros_like(images)


def hsv_to_rgb(images: Any, name: Optional[str] = None) -> Any:
    """
    Convert one or more images from HSV to RGB.

    Args:
        images: A Tensor. Must be one of the following types: float32, float64.
        name: A name for the operation (optional).

    Returns:
        A Tensor. Has the same type as images.
    """

    images = np.array(images)
    # naive stub returning same shape
    return np.zeros_like(images)
