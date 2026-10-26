"""Unit tests for `zero_tensorflow.image`."""

from __future__ import annotations

import zero_tensorflow as tf
from zero_tensorflow import Tensor, debugging, image
from zero_tensorflow._typing import TensorOperand

import pytest
from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_ir import LogicalGraph


def test_image_resize() -> None:
    """Verify `image.resize` executes real resizing and honors `preserve_aspect_ratio`."""
    img = tf.fill((4, 8, 3), 3.0)
    res = image.resize(img, [2, 2])
    assert isinstance(res, Tensor)
    assert res.shape == (2, 2, 3)
    debugging.assert_near(res, tf.fill((2, 2, 3), 3.0))

    preserved = image.resize(img, [2, 2], preserve_aspect_ratio=True)
    assert preserved.shape == (1, 2, 3)
    debugging.assert_near(preserved, tf.fill((1, 2, 3), 3.0))

    nearest_3d = image.resize(img, [2, 2], method="nearest")
    assert nearest_3d.shape == (2, 2, 3)
    resized_4d = image.resize(tf.ones((2, 4, 4, 3)), [2, 2])
    assert resized_4d.shape == (2, 2, 2, 3)
    resized_2d = image.resize(tf.ones((4, 4)), [2, 2])
    assert resized_2d.shape == (2, 2)


def test_image_crop_and_resize() -> None:
    """Verify `image.crop_and_resize` with bilinear, nearest, and extrapolation_value."""
    img = tf.reshape(tf.cast(tf.range(0, 16, 1), tf.float32), (1, 4, 4, 1))
    boxes = Tensor([[0.0, 0.0, 1.0, 1.0], [-1.0, -1.0, -0.5, -0.5]], dtype=tf.float32)
    box_ind = Tensor([0, 0], dtype=tf.int32)
    res_bilinear = image.crop_and_resize(
        img,
        boxes,
        box_ind,
        [2, 2],
        method="bilinear",
        extrapolation_value=-9.0,
    )
    assert isinstance(res_bilinear, Tensor)
    assert res_bilinear.shape == (2, 2, 2, 1)
    debugging.assert_near(
        tf.reshape(res_bilinear[0], (2, 2)),
        [[0.0, 3.0], [12.0, 15.0]],
    )
    debugging.assert_near(
        tf.reshape(res_bilinear[1], (2, 2)),
        [[-9.0, -9.0], [-9.0, -9.0]],
    )

    res_nearest = image.crop_and_resize(
        img,
        [[0.0, 0.0, 1.0, 1.0], [0.0]],
        [0],
        [1, 1],
        method="nearest",
    )
    assert res_nearest.shape == (1, 1, 1, 1)


def test_image_non_max_suppression() -> None:
    """Verify `image.non_max_suppression` and `image.non_max_suppression_padded`."""
    boxes = [
        [0.0, 0.0, 1.0, 1.0],
        [0.05, 0.05, 0.95, 0.95],
        [10.0, 10.0, 11.0, 11.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    scores = [0.9, 0.85, 0.7, 0.95]
    res = image.non_max_suppression(boxes, scores, max_output_size=2, iou_threshold=0.5)
    debugging.assert_equal(res, [3, 0])

    res2 = image.non_max_suppression(
        boxes[:3],
        scores[:3],
        max_output_size=2,
        iou_threshold=0.5,
        score_threshold=0.5,
    )
    debugging.assert_equal(res2, [0, 2])

    padded_idx, valid_cnt = image.non_max_suppression_padded(
        boxes[:2],
        scores[:2],
        max_output_size=3,
        iou_threshold=0.5,
        pad_to_max_output_size=True,
    )
    assert valid_cnt.numpy() == 1
    debugging.assert_equal(padded_idx, [0, 0, 0])

    unpadded_idx, valid_cnt2 = image.non_max_suppression_padded(
        boxes[:2],
        scores[:2],
        max_output_size=1,
        pad_to_max_output_size=False,
    )
    assert valid_cnt2.numpy() == 1
    debugging.assert_equal(unpadded_idx, [0])


def test_image_rgb_to_hsv() -> None:
    """Verify `image.rgb_to_hsv` computes expected HSV values."""
    img = Tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=tf.float32)
    res = image.rgb_to_hsv(img)
    assert isinstance(res, Tensor)
    assert res.shape == (1, 2, 3)
    debugging.assert_near(
        res,
        [[[0.0, 1.0, 1.0], [1.0 / 3.0, 1.0, 1.0]]],
        atol=1e-5,
    )


def test_image_hsv_to_rgb() -> None:
    """Verify `image.hsv_to_rgb` round-trips HSV back to RGB."""
    hsv = Tensor([[[0.0, 1.0, 1.0], [1.0 / 3.0, 1.0, 1.0]]], dtype=tf.float32)
    res = image.hsv_to_rgb(hsv)
    assert isinstance(res, Tensor)
    assert res.shape == (1, 2, 3)
    debugging.assert_near(
        res,
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        atol=1e-5,
    )


def test_image_rgb_to_grayscale() -> None:
    """Verify `image.rgb_to_grayscale` computes luminance grayscale."""
    img = tf.ones((2, 2, 3), dtype=tf.float32)
    res = image.rgb_to_grayscale(img)
    assert isinstance(res, Tensor)
    assert res.shape == (2, 2, 1)
    debugging.assert_near(res, tf.ones((2, 2, 1), dtype=tf.float32), atol=1e-3)


def test_image_yiq_to_rgb() -> None:
    """Verify `image.rgb_to_yiq` and `image.yiq_to_rgb` round-trip accurately."""
    rgb = Tensor([[[0.5, 0.25, 0.75], [0.1, 0.8, 0.3]]], dtype=tf.float32)
    yiq = image.rgb_to_yiq(rgb)
    assert isinstance(yiq, Tensor)
    assert yiq.shape == (1, 2, 3)
    recovered = image.yiq_to_rgb(yiq)
    assert isinstance(recovered, Tensor)
    assert recovered.shape == (1, 2, 3)
    debugging.assert_near(recovered, rgb, atol=1e-5)


def test_image_rgb_to_yiq() -> None:
    """Verify `image.rgb_to_yiq` computes NTSC YIQ components."""
    rgb = tf.ones((1, 1, 3), dtype=tf.float32)
    yiq = image.rgb_to_yiq(rgb)
    assert isinstance(yiq, Tensor)
    debugging.assert_near(yiq, [[[1.0, 0.0, 0.0]]], atol=1e-5)


def test_image_stubs() -> None:
    """Verify `extract_patches`, `resize_with_crop_or_pad`, `crop_to_bounding_box`, and random augmentations."""
    with pytest.raises(NotImplementedError):
        image.non_max_suppression_padded()
    with pytest.raises(NotImplementedError):
        image.extract_patches()

    src_patch_img = tf.reshape(tf.cast(tf.range(0, 16, 1), tf.float32), (1, 4, 4, 1))
    patches = image.extract_patches(
        src_patch_img,
        sizes=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
    )
    assert patches.shape == (1, 2, 2, 4)
    debugging.assert_near(
        patches[0][0][0],
        [0.0, 1.0, 4.0, 5.0],
    )
    patches_same = image.extract_patches(
        src_patch_img,
        sizes=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        rates=[1, 1, 1, 1],
        padding="SAME",
    )
    assert patches_same.shape == (1, 2, 2, 9)

    img3 = tf.fill((4, 4, 3), 5.0)
    img4 = tf.fill((1, 4, 4, 3), 7.0)
    cropped3 = image.resize_with_crop_or_pad(img3, 2, 2)
    assert cropped3.shape == (2, 2, 3)
    debugging.assert_near(cropped3, tf.fill((2, 2, 3), 5.0))

    padded3 = image.resize_with_crop_or_pad(img3, 6, 6)
    assert padded3.shape == (6, 6, 3)
    assert padded3[2][2][0].numpy() == 5.0
    assert padded3[0][0][0].numpy() == 0.0

    same3 = image.resize_with_crop_or_pad(img3, 4, 4)
    assert same3.shape == (4, 4, 3)
    asym3 = image.resize_with_crop_or_pad(tf.ones((2, 2, 1), dtype=tf.float32), 3, 2)
    assert asym3.shape == (3, 2, 1)

    cropped4 = image.resize_with_crop_or_pad(img4, 2, 2)
    assert cropped4.shape == (1, 2, 2, 3)
    debugging.assert_near(cropped4, tf.fill((1, 2, 2, 3), 7.0))
    assert image.resize_with_crop_or_pad(None, 2, 2).shape == (2, 2, 1)
    assert image.resize_with_crop_or_pad([1.0, 2.0], 2, 2).shape == (2, 2, 1)

    assert image.crop_to_bounding_box(img3, 0, 0, 2, 2).shape == (2, 2, 3)
    assert image.crop_to_bounding_box(img4, 0, 0, 2, 2).shape == (1, 2, 2, 3)
    assert image.crop_to_bounding_box(None, 0, 0, 2, 2).shape == (2, 2, 1)
    assert image.crop_to_bounding_box([1.0, 2.0], 0, 0, 2, 2).shape == (2, 2, 1)

    assert image.random_brightness(img3, 0.2, seed=42).shape == (4, 4, 3)
    assert image.random_brightness(None).shape == (1,)
    with pytest.raises(ValueError):
        image.random_brightness(img3, -0.1)

    assert image.random_contrast(img3, 0.5, 1.5, seed=42).shape == (4, 4, 3)
    assert image.random_contrast(None).shape == (1,)
    with pytest.raises(ValueError):
        image.random_contrast(img3, -0.5, 1.5)
    with pytest.raises(ValueError):
        image.random_contrast(img3, 1.5, 0.5)

    assert image.random_flip_left_right(img3, seed=42).shape == (4, 4, 3)
    assert image.random_flip_left_right(img3, seed=1).shape == (4, 4, 3)
    assert image.random_flip_left_right(tf.ones((2, 2), dtype=tf.float32), seed=1).shape == (2, 2)
    assert image.random_flip_left_right(None).shape == (1,)


def test_ensure_tensor_and_tracing() -> None:
    """Verify `image._ensure_tensor`, helper conversions, and traced image ops."""
    t: TensorOperand = image._ensure_tensor(1)
    assert t is not None
    assert image._to_py_nested("non_num") == 0.0
    assert image._as_float("non_num") == 0.0
    assert image._map_color_channels(1.0, ("rgb_to_hsv",)) == 1.0
    assert image.non_max_suppression([[0.0]], [0.9], 1).shape == (0,)

    prev_eager = _compiler_config.eager_mode
    prev_tracing = _tracer.is_tracing
    prev_graph = _tracer.active_graph
    _compiler_config.eager_mode = False
    _tracer.is_tracing = True
    _tracer.active_graph = LogicalGraph(name="img_trace")
    try:
        pt4 = tf._to_tensor(ProxyTensor(id="img4", shape=(1, 4, 4, 3), dtype="float32"))
        pt_boxes = tf._to_tensor(ProxyTensor(id="boxes", shape=(1, 4), dtype="float32"))
        pt_ind = tf._to_tensor(ProxyTensor(id="ind", shape=(1,), dtype="int32"))
        pt_scores = tf._to_tensor(ProxyTensor(id="scores", shape=(1,), dtype="float32"))
        assert isinstance(image.resize(pt4, [2, 2]), Tensor)
        assert isinstance(image.crop_and_resize(pt4, pt_boxes, pt_ind, [2, 2]), Tensor)
        assert isinstance(image.non_max_suppression(pt_boxes, pt_scores, 1), Tensor)
        assert isinstance(image.rgb_to_hsv(pt4), Tensor)
        assert isinstance(image.hsv_to_rgb(pt4), Tensor)
        assert isinstance(image.rgb_to_yiq(pt4), Tensor)
        assert isinstance(image.yiq_to_rgb(pt4), Tensor)
    finally:
        _compiler_config.eager_mode = prev_eager
        _tracer.is_tracing = prev_tracing
        _tracer.active_graph = prev_graph

    # Test ResizeMethod and newly added v0.0.5 tf.image operations
    assert image.ResizeMethod.BILINEAR == "bilinear"
    assert image.ResizeMethod.NEAREST_NEIGHBOR == "nearest"
    img3 = tf.ones((2, 2, 3), dtype=tf.float32)
    img4 = tf.ones((1, 2, 2, 3), dtype=tf.float32)
    debugging.assert_near(image.adjust_brightness(Tensor([1.0]), 0.5), [1.5])
    assert image.adjust_contrast(img3, 2.0).shape == (2, 2, 3)
    debugging.assert_near(image.adjust_gamma(Tensor([4.0]), gamma=0.5, gain=2.0), [4.0])
    assert image.adjust_hue(img3, 0.1).shape == (2, 2, 3)
    assert image.adjust_jpeg_quality(img3, 80).shape == (2, 2, 3)
    assert image.adjust_saturation(img3, 1.2).shape == (2, 2, 3)
    assert image.central_crop(img3, 0.5).shape == (2, 2, 3)
    assert len(image.combined_non_max_suppression(img4, img3, 1, 2)) == 4
    assert image.convert_image_dtype(img3, "float32").dtype == tf.float32
    assert image.decode_and_crop_jpeg(b"abc", [0, 0, 1, 1], channels=3).shape == (1, 1, 3)
    assert image.decode_bmp(b"bmp", channels=1).shape == (1, 1, 1)
    assert image.decode_gif(b"gif").shape == (1, 1, 1, 3)
    assert image.decode_image(b"img", channels=3).shape == (1, 1, 3)
    assert image.decode_jpeg(b"jpg").shape == (1, 1, 3)
    assert image.decode_jxl(b"jxl").shape == (1, 1, 3)
    assert image.decode_png(b"png").shape == (1, 1, 3)
    assert image.decode_webp(b"webp").shape == (1, 1, 3)
    assert image.draw_bounding_boxes(img4, [[[0.0, 0.0, 1.0, 1.0]]], [[1.0, 0.0, 0.0]]).shape == (
        1,
        2,
        2,
        3,
    )
    assert isinstance(image.encode_jpeg(img3), Tensor)
    assert isinstance(image.encode_png(img3), Tensor)
    assert image.extract_glimpse(img4, [1, 1], [[0.0, 0.0]]).shape == (1, 2, 2, 3)
    debugging.assert_equal(image.extract_jpeg_shape(b"jpg"), [1, 1, 3])
    assert image.flip_left_right(img3).shape == (2, 2, 3)
    assert image.flip_left_right(Tensor([1.0, 2.0])).shape == (2,)
    assert image.flip_up_down(img3).shape == (2, 2, 3)
    assert image.flip_up_down(Tensor([[1.0], [2.0]])).shape == (2, 1)
    assert len(image.generate_bounding_box_proposals(img4, img4, img3, img3, post_nms_topn=2)) == 2
    assert image.grayscale_to_rgb(tf.ones((2, 2, 1), dtype=tf.float32)).shape == (2, 2, 3)
    dy, dx = image.image_gradients(img4)
    assert dy.shape == (1, 2, 2, 3) and dx.shape == (1, 2, 2, 3)
    assert bool(image.is_jpeg(b"\xff\xd8\xff")) is True
    assert bool(image.is_jpeg("\xff\xd8")) is True
    assert bool(image.is_jpeg(b"png")) is False
    assert image.non_max_suppression_overlaps([[1.0]], [0.9], 1).shape == (1,)
    sel_i, sel_s = image.non_max_suppression_with_scores([[0.0, 0.0, 1.0, 1.0]], [0.9], 1)
    assert sel_i.shape == (1,) and sel_s.shape == (1,)
    assert image.pad_to_bounding_box(img3, 0, 0, 4, 4).shape == (4, 4, 3)
    assert image.per_image_standardization(img3).shape == (2, 2, 3)
    debugging.assert_near(image.psnr(img3, img3, max_val=1.0), 40.0)
    assert image.random_crop(img3, (1, 1, 3)).shape == (1, 1, 3)
    assert image.random_flip_up_down(img3, seed=1).shape == (2, 2, 3)
    assert image.random_hue(img3, 0.1, seed=1).shape == (2, 2, 3)
    assert image.random_jpeg_quality(img3, 70, 90, seed=1).shape == (2, 2, 3)
    assert image.random_saturation(img3, 0.5, 1.5, seed=1).shape == (2, 2, 3)
    assert image.resize_with_pad(img3, 4, 4).shape == (4, 4, 3)
    assert image.rgb_to_yuv(img3).shape == (2, 2, 3)
    assert image.yuv_to_rgb(img3).shape == (2, 2, 3)
    assert image.rot90(img3, k=1).shape == (2, 2, 3)
    assert len(image.sample_distorted_bounding_box((2, 2, 3), [[[0.0, 0.0, 1.0, 1.0]]])) == 3
    assert (
        len(
            image.stateless_sample_distorted_bounding_box(
                (2, 2, 3), [[[0.0, 0.0, 1.0, 1.0]]], seed=[1, 2]
            )
        )
        == 3
    )
    assert image.sobel_edges(img4).shape == (1, 2, 2, 3, 2)
    debugging.assert_near(image.ssim(img4, img4, max_val=1.0), 1.0)
    debugging.assert_near(image.ssim_multiscale(img4, img4, max_val=1.0), 1.0)
    assert image.stateless_random_brightness(img3, 0.1, [1, 2]).shape == (2, 2, 3)
    assert image.stateless_random_contrast(img3, 0.5, 1.5, [1, 2]).shape == (2, 2, 3)
    assert image.stateless_random_crop(img3, (1, 1, 3), [1, 2]).shape == (1, 1, 3)
    assert image.stateless_random_flip_left_right(img3, [1, 2]).shape == (2, 2, 3)
    assert image.stateless_random_flip_up_down(img3, [1, 2]).shape == (2, 2, 3)
    assert image.stateless_random_hue(img3, 0.1, [1, 2]).shape == (2, 2, 3)
    assert image.stateless_random_jpeg_quality(img3, 70, 90, [1, 2]).shape == (2, 2, 3)
    assert image.stateless_random_saturation(img3, 0.5, 1.5, [1, 2]).shape == (2, 2, 3)
    debugging.assert_near(image.total_variation(img4), 0.0)
    assert image.transpose(tf.ones((1, 2, 3, 4), dtype=tf.float32)).shape == (1, 3, 2, 4)
    assert image.transpose(tf.ones((2, 3, 4), dtype=tf.float32)).shape == (3, 2, 4)
    assert image.transpose(Tensor([1.0, 2.0])).shape == (2,)
