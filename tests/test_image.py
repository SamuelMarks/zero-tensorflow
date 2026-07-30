from unittest import mock

import numpy as np
import pytest

from zero_tensorflow import image


@mock.patch("zero_tensorflow.image._ops.resize", return_value=np.zeros((1, 5, 5, 3)))
def test_image_resize(mock_resize):
    res = image.resize(np.zeros((1, 10, 10, 3)), [5, 5])
    assert res is not None


@mock.patch(
    "zero_tensorflow.image._ops.crop_and_resize", return_value=np.zeros((2, 5, 5, 3))
)
def test_image_crop_and_resize(mock_crop):
    res = image.crop_and_resize(
        np.zeros((1, 10, 10, 3)), np.zeros((2, 4)), np.zeros((2,)), [5, 5]
    )
    assert res is not None


@mock.patch(
    "zero_tensorflow.image._ops.non_max_suppression", return_value=np.zeros((1,))
)
def test_image_non_max_suppression(mock_nms):
    res = image.non_max_suppression(np.zeros((2, 4)), np.zeros((2,)), 1)
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.rgb_to_hsv", return_value=np.zeros((1, 1, 3)))
def test_image_rgb_to_hsv(mock_rgb):
    res = image.rgb_to_hsv(np.zeros((1, 1, 3)))
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.hsv_to_rgb", return_value=np.zeros((1, 1, 3)))
def test_image_hsv_to_rgb(mock_hsv):
    res = image.hsv_to_rgb(np.zeros((1, 1, 3)))
    assert res is not None


@mock.patch(
    "zero_tensorflow.image._ops.rgb_to_grayscale", return_value=np.zeros((1, 1, 1))
)
def test_image_rgb_to_grayscale(mock_gray):
    res = image.rgb_to_grayscale(np.zeros((1, 1, 3)))
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.yiq_to_rgb", return_value=np.zeros((1, 1, 3)))
def test_image_yiq_to_rgb(mock_yiq):
    res = image.yiq_to_rgb(np.zeros((1, 1, 3)))
    assert res is not None


@mock.patch("zero_tensorflow.image._ops.rgb_to_yiq", return_value=np.zeros((1, 1, 3)))
def test_image_rgb_to_yiq(mock_rgb_yiq):
    res = image.rgb_to_yiq(np.zeros((1, 1, 3)))
    assert res is not None


def test_not_implemented():
    with pytest.raises(NotImplementedError):
        image.non_max_suppression_padded()
    with pytest.raises(NotImplementedError):
        image.resize_with_crop_or_pad()
    with pytest.raises(NotImplementedError):
        image.crop_to_bounding_box()
    with pytest.raises(NotImplementedError):
        image.extract_patches()
    with pytest.raises(NotImplementedError):
        image.random_brightness()
    with pytest.raises(NotImplementedError):
        image.random_contrast()
    with pytest.raises(NotImplementedError):
        image.random_flip_left_right()


def test_ensure_tensor():
    assert image._ensure_tensor(1) is not None
