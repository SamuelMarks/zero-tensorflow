"""Unit tests for `test_image`."""

from __future__ import annotations

from unittest import mock

from zero_tensorflow import image
from zero_tensorflow._typing import TensorOperand

import numpy as np
import pytest


@mock.patch("zero_tensorflow.image._ops.resize", return_value=np.zeros((1, 5, 5, 3)))
def test_image_resize(mock_resize: TensorOperand) -> None:
    """Execute `test_image_resize`.

    Args:
        mock_resize: Parameter `mock_resize`.
    """
    res = image.resize(np.zeros((1, 10, 10, 3)), [5, 5])
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.crop_and_resize", return_value=np.zeros((2, 5, 5, 3)))
def test_image_crop_and_resize(mock_crop: TensorOperand) -> None:
    """Execute `test_image_crop_and_resize`.

    Args:
        mock_crop: Parameter `mock_crop`.
    """
    res = image.crop_and_resize(np.zeros((1, 10, 10, 3)), np.zeros((2, 4)), np.zeros((2,)), [5, 5])
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.non_max_suppression", return_value=np.zeros((1,)))
def test_image_non_max_suppression(mock_nms: TensorOperand) -> None:
    """Execute `test_image_non_max_suppression`.

    Args:
        mock_nms: Parameter `mock_nms`.
    """
    res = image.non_max_suppression(np.zeros((2, 4)), np.zeros((2,)), 1)
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.rgb_to_hsv", return_value=np.zeros((1, 1, 3)))
def test_image_rgb_to_hsv(mock_rgb: TensorOperand) -> None:
    """Execute `test_image_rgb_to_hsv`.

    Args:
        mock_rgb: Parameter `mock_rgb`.
    """
    res = image.rgb_to_hsv(np.zeros((1, 1, 3)))
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.hsv_to_rgb", return_value=np.zeros((1, 1, 3)))
def test_image_hsv_to_rgb(mock_hsv: TensorOperand) -> None:
    """Execute `test_image_hsv_to_rgb`.

    Args:
        mock_hsv: Parameter `mock_hsv`.
    """
    res = image.hsv_to_rgb(np.zeros((1, 1, 3)))
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.rgb_to_grayscale", return_value=np.zeros((1, 1, 1)))
def test_image_rgb_to_grayscale(mock_gray: TensorOperand) -> None:
    """Execute `test_image_rgb_to_grayscale`.

    Args:
        mock_gray: Parameter `mock_gray`.
    """
    res = image.rgb_to_grayscale(np.zeros((1, 1, 3)))
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.yiq_to_rgb", return_value=np.zeros((1, 1, 3)))
def test_image_yiq_to_rgb(mock_yiq: TensorOperand) -> None:
    """Execute `test_image_yiq_to_rgb`.

    Args:
        mock_yiq: Parameter `mock_yiq`.
    """
    res = image.yiq_to_rgb(np.zeros((1, 1, 3)))
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.rgb_to_yiq", return_value=np.zeros((1, 1, 3)))
def test_image_rgb_to_yiq(mock_rgb_yiq: TensorOperand) -> None:
    """Execute `test_image_rgb_to_yiq`.

    Args:
        mock_rgb_yiq: Parameter `mock_rgb_yiq`.
    """
    res = image.rgb_to_yiq(np.zeros((1, 1, 3)))
    assert res is not None


def test_not_implemented() -> None:
    """Verify `image` operations and `NotImplementedError` for unimplemented ops."""
    with pytest.raises(NotImplementedError):
        image.non_max_suppression_padded()
    with pytest.raises(NotImplementedError):
        image.extract_patches()

    img3 = np.ones((4, 4, 3), dtype=np.float32)
    img4 = np.ones((2, 4, 4, 3), dtype=np.float32)
    assert image.resize_with_crop_or_pad(img3, 2, 2).shape == (2, 2, 3)
    assert image.resize_with_crop_or_pad(img4, 2, 2).shape == (2, 2, 2, 3)
    assert image.resize_with_crop_or_pad(None, 2, 2).shape == (2, 2, 1)

    assert image.crop_to_bounding_box(img3, 1, 1, 2, 2).shape == (2, 2, 3)
    assert image.crop_to_bounding_box(img4, 1, 1, 2, 2).shape == (2, 2, 2, 3)
    assert image.crop_to_bounding_box(np.ones((4, 4)), 0, 0, 2, 2).shape == (2, 2, 1)
    assert image.crop_to_bounding_box(None, 0, 0, 2, 2).shape == (2, 2, 1)

    assert image.random_brightness(img3, 0.2, seed=42).shape == (4, 4, 3)
    assert image.random_brightness(None, 0.2).shape == (1,)
    with pytest.raises(ValueError, match="non-negative"):
        image.random_brightness(img3, -0.1)

    assert image.random_contrast(img3, 0.5, 1.5, seed=42).shape == (4, 4, 3)
    assert image.random_contrast(None, 0.5, 1.5).shape == (1,)
    with pytest.raises(ValueError, match="lower"):
        image.random_contrast(img3, -0.5, 1.5)

    assert image.random_flip_left_right(img3, seed=1).shape == (4, 4, 3)
    assert image.random_flip_left_right(img3, seed=0).shape == (4, 4, 3)
    assert image.random_flip_left_right(None).shape == (1,)


def test_ensure_tensor() -> None:
    """Execute `test_ensure_tensor`."""
    assert image._ensure_tensor(1) is not None
