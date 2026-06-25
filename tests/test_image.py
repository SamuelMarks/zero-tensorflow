import numpy as np
from zero_tensorflow import image


def test_image_resize():
    # 3D
    img3d = np.ones((10, 10, 3))
    res3d = image.resize(img3d, [5, 5])
    assert res3d.shape == (5, 5, 3)

    # 4D
    img4d = np.ones((2, 10, 10, 3))
    res4d = image.resize(img4d, [5, 5])
    assert res4d.shape == (2, 5, 5, 3)

    # 2D fallback
    img2d = np.ones((10, 10))
    res2d = image.resize(img2d, [5, 5])
    assert res2d.shape == (10, 10)


def test_image_crop_and_resize():
    img = np.ones((1, 10, 10, 3))
    boxes = [[0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9]]
    box_indices = [0, 0]
    crop_size = [5, 5]
    res = image.crop_and_resize(img, boxes, box_indices, crop_size)
    assert res.shape == (2, 5, 5, 3)


def test_image_non_max_suppression():
    boxes = [[0, 0, 1, 1], [0, 0, 1, 1]]
    scores = [0.9, 0.1]
    res = image.non_max_suppression(boxes, scores, 1)
    assert list(res) == [0]


def test_image_rgb_to_hsv():
    img = np.ones((10, 10, 3))
    res = image.rgb_to_hsv(img)
    assert res.shape == (10, 10, 3)


def test_image_hsv_to_rgb():
    img = np.ones((10, 10, 3))
    res = image.hsv_to_rgb(img)
    assert res.shape == (10, 10, 3)
