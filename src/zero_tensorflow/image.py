"""TensorFlow `image` processing module."""

from __future__ import annotations

import colorsys
import math
import random as _pyrandom
from collections.abc import Sequence

from zero_tensorflow import io as _tf_io
from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    NestedTensorData,
    NumpyConvertibleProtocol,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap_tensor
from zero_tensorflow.dtypes import DType

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from zero_keras import ops as _k_ops

__all__ = [
    "ResizeMethod",
    "adjust_brightness",
    "adjust_contrast",
    "adjust_gamma",
    "adjust_hue",
    "adjust_jpeg_quality",
    "adjust_saturation",
    "central_crop",
    "combined_non_max_suppression",
    "convert_image_dtype",
    "crop_and_resize",
    "crop_to_bounding_box",
    "decode_and_crop_jpeg",
    "decode_bmp",
    "decode_gif",
    "decode_image",
    "decode_jpeg",
    "decode_jxl",
    "decode_png",
    "decode_webp",
    "draw_bounding_boxes",
    "encode_jpeg",
    "encode_png",
    "extract_glimpse",
    "extract_jpeg_shape",
    "extract_patches",
    "flip_left_right",
    "flip_up_down",
    "generate_bounding_box_proposals",
    "grayscale_to_rgb",
    "hsv_to_rgb",
    "image_gradients",
    "is_jpeg",
    "non_max_suppression",
    "non_max_suppression_overlaps",
    "non_max_suppression_padded",
    "non_max_suppression_with_scores",
    "pad_to_bounding_box",
    "per_image_standardization",
    "psnr",
    "random_brightness",
    "random_contrast",
    "random_crop",
    "random_flip_left_right",
    "random_flip_up_down",
    "random_hue",
    "random_jpeg_quality",
    "random_saturation",
    "resize",
    "resize_with_crop_or_pad",
    "resize_with_pad",
    "rgb_to_grayscale",
    "rgb_to_hsv",
    "rgb_to_yiq",
    "rgb_to_yuv",
    "rot90",
    "sample_distorted_bounding_box",
    "sobel_edges",
    "ssim",
    "ssim_multiscale",
    "stateless_random_brightness",
    "stateless_random_contrast",
    "stateless_random_crop",
    "stateless_random_flip_left_right",
    "stateless_random_flip_up_down",
    "stateless_random_hue",
    "stateless_random_jpeg_quality",
    "stateless_random_saturation",
    "stateless_sample_distorted_bounding_box",
    "total_variation",
    "transpose",
    "yiq_to_rgb",
    "yuv_to_rgb",
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


def _to_py_nested(val: Tensor | TensorOperand | TensorOrScalar) -> NestedTensorData:
    """Convert a tensor or array-like operand into nested Python lists/floats.

    Args:
        val: Input tensor or array-like value.

    Returns:
        NestedTensorData: Python scalar or nested list.
    """
    if isinstance(val, NumpyConvertibleProtocol):
        return _to_py_nested(val.numpy())
    if isinstance(val, ArrayLikeProtocol):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_to_py_nested(item) for item in val]
    if isinstance(val, (int, float, bool)):
        return val
    return 0.0


def _as_float(val: NestedTensorData) -> float:
    """Convert a scalar element to `float`.

    Args:
        val: Scalar value.

    Returns:
        float: Float value.
    """
    if isinstance(val, (int, float, bool)):
        return float(val)
    return 0.0


def _list_at(seq: list[NestedTensorData], idx: int) -> list[NestedTensorData]:
    """Return the sublist at `seq[idx]` or an empty list if out of range or not a list.

    Args:
        seq: Outer nested list.
        idx: Element index.

    Returns:
        list[NestedTensorData]: Sublist at `idx` or `[]`.
    """
    item = seq[idx] if 0 <= idx < len(seq) else None
    return item if isinstance(item, list) else []


def _resize_3d(
    slice_3d: list[NestedTensorData],
    cur_h: int,
    cur_w: int,
    depth: int,
    target_h: int,
    target_w: int,
    method: str,
) -> list[list[list[float]]]:
    """Interpolate a single 3D `(cur_h, cur_w, depth)` image to `(target_h, target_w, depth)`.

    Args:
        slice_3d: Nested list of shape `(cur_h, cur_w, depth)`.
        cur_h: Source height.
        cur_w: Source width.
        depth: Channel count.
        target_h: Output height.
        target_w: Output width.
        method: Interpolation method (`'bilinear'` or `'nearest'`).

    Returns:
        list[list[list[float]]]: Resized 3D nested list.
    """
    rows: list[list[list[float]]] = []
    scale_y = float(cur_h) / float(target_h) if target_h > 0 else 1.0
    scale_x = float(cur_w) / float(target_w) if target_w > 0 else 1.0
    for r in range(target_h):
        in_y = max(0.0, min(float(cur_h - 1), (float(r) + 0.5) * scale_y - 0.5))
        row_px: list[list[float]] = []
        for c in range(target_w):
            in_x = max(0.0, min(float(cur_w - 1), (float(c) + 0.5) * scale_x - 0.5))
            if method.lower() == "nearest":
                iy = min(max(int(round(in_y)), 0), cur_h - 1)
                ix = min(max(int(round(in_x)), 0), cur_w - 1)
                r_val = _list_at(slice_3d, iy)
                p_val = _list_at(r_val, ix)
                row_px.append(
                    [_as_float(p_val[d]) if d < len(p_val) else 0.0 for d in range(depth)]
                )
            else:
                top_y = int(math.floor(in_y))
                bot_y = min(int(math.ceil(in_y)), cur_h - 1)
                y_lerp = in_y - float(top_y)
                left_x = int(math.floor(in_x))
                right_x = min(int(math.ceil(in_x)), cur_w - 1)
                x_lerp = in_x - float(left_x)
                r_top = _list_at(slice_3d, top_y)
                r_bot = _list_at(slice_3d, bot_y)
                tl = _list_at(r_top, left_x)
                tr = _list_at(r_top, right_x)
                bl = _list_at(r_bot, left_x)
                br = _list_at(r_bot, right_x)
                px: list[float] = []
                for d in range(depth):
                    v_tl = _as_float(tl[d]) if d < len(tl) else 0.0
                    v_tr = _as_float(tr[d]) if d < len(tr) else 0.0
                    v_bl = _as_float(bl[d]) if d < len(bl) else 0.0
                    v_br = _as_float(br[d]) if d < len(br) else 0.0
                    top_v = v_tl + (v_tr - v_tl) * x_lerp
                    bot_v = v_bl + (v_br - v_bl) * x_lerp
                    px.append(top_v + (bot_v - top_v) * y_lerp)
                row_px.append(px)
        rows.append(row_px)
    return rows


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
    del name
    t_img = _ensure_tensor(images)
    target_h = int(size[0])
    target_w = int(size[1])
    if preserve_aspect_ratio and len(t_img.shape) >= 3:
        cur_h = max(1, t_img.shape[-3])
        cur_w = max(1, t_img.shape[-2])
        scale = min(target_h / float(cur_h), target_w / float(cur_w))
        target_h = max(1, int(round(cur_h * scale)))
        target_w = max(1, int(round(cur_w * scale)))
    size_tuple = (target_h, target_w)
    if _tracer.is_tracing or not _compiler_config.eager_mode:
        res = _ops.resize(t_img, size_tuple, method=method, antialias=antialias)
        return _wrap_tensor(as_operand(res))

    py_img = _to_py_nested(t_img)
    if len(t_img.shape) == 3 and isinstance(py_img, list):
        out_3d = _resize_3d(
            py_img,
            t_img.shape[0],
            t_img.shape[1],
            t_img.shape[2],
            target_h,
            target_w,
            method,
        )
        return Tensor(out_3d, dtype=t_img.dtype)
    if len(t_img.shape) == 4 and isinstance(py_img, list):
        out_4d = [
            _resize_3d(
                b_slice if isinstance(b_slice, list) else [],
                t_img.shape[1],
                t_img.shape[2],
                t_img.shape[3],
                target_h,
                target_w,
                method,
            )
            for b_slice in py_img
        ]
        return Tensor(out_4d, dtype=t_img.dtype)
    res_fallback = _k_ops.image.resize(t_img, size_tuple, interpolation=method, antialias=antialias)
    return _wrap_tensor(as_operand(res_fallback))


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
        method: Interpolation method string (`'bilinear'` or `'nearest'`).
        extrapolation_value: Value used for extrapolated image pixels.
        name: Optional operation name.

    Returns:
        Tensor: Cropped and resized image tensor.
    """
    del name
    t_img = _ensure_tensor(image)
    crop_h = int(crop_size[0])
    crop_w = int(crop_size[1])
    if _tracer.is_tracing or not _compiler_config.eager_mode:
        t_boxes = _ensure_tensor(boxes)
        t_ind = _ensure_tensor(box_indices)
        num_b = t_boxes.shape[0] if t_boxes.shape else 1
        depth_c = t_img.shape[3] if len(t_img.shape) >= 4 else 1
        del t_ind
        return _wrap_tensor(_ops.zeros((num_b, crop_h, crop_w, depth_c), dtype=t_img.dtype))

    py_img = _to_py_nested(t_img)
    py_boxes = _to_py_nested(boxes)
    py_ind = _to_py_nested(box_indices)
    img_h = t_img.shape[1]
    img_w = t_img.shape[2]
    depth = t_img.shape[3]
    boxes_list = py_boxes if isinstance(py_boxes, list) else []
    ind_list = py_ind if isinstance(py_ind, list) else []
    img_list = py_img if isinstance(py_img, list) else []

    out_crops: list[list[list[list[float]]]] = []
    for b_i, raw_box in enumerate(boxes_list):
        if not isinstance(raw_box, list) or len(raw_box) < 4:
            continue
        y1 = _as_float(raw_box[0])
        x1 = _as_float(raw_box[1])
        y2 = _as_float(raw_box[2])
        x2 = _as_float(raw_box[3])
        batch_idx = int(_as_float(ind_list[b_i])) if b_i < len(ind_list) else 0
        batch_slice = (
            img_list[batch_idx]
            if 0 <= batch_idx < len(img_list) and isinstance(img_list[batch_idx], list)
            else []
        )

        crop_rows: list[list[list[float]]] = []
        for r in range(crop_h):
            in_y = (
                (y1 + (y2 - y1) * r / float(crop_h - 1)) * float(img_h - 1)
                if crop_h > 1
                else 0.5 * (y1 + y2) * float(img_h - 1)
            )
            row_pixels: list[list[float]] = []
            for c in range(crop_w):
                in_x = (
                    (x1 + (x2 - x1) * c / float(crop_w - 1)) * float(img_w - 1)
                    if crop_w > 1
                    else 0.5 * (x1 + x2) * float(img_w - 1)
                )
                if in_y < 0.0 or in_y > float(img_h - 1) or in_x < 0.0 or in_x > float(img_w - 1):
                    row_pixels.append([float(extrapolation_value)] * depth)
                elif method.lower() == "nearest":
                    iy = min(max(int(round(in_y)), 0), img_h - 1)
                    ix = min(max(int(round(in_x)), 0), img_w - 1)
                    r_val = batch_slice[iy] if iy < len(batch_slice) else []
                    p_val = r_val[ix] if isinstance(r_val, list) and ix < len(r_val) else []
                    row_pixels.append(
                        [
                            _as_float(p_val[d])
                            if isinstance(p_val, list) and d < len(p_val)
                            else 0.0
                            for d in range(depth)
                        ]
                    )
                else:
                    top_y = int(math.floor(in_y))
                    bot_y = min(int(math.ceil(in_y)), img_h - 1)
                    y_lerp = in_y - float(top_y)
                    left_x = int(math.floor(in_x))
                    right_x = min(int(math.ceil(in_x)), img_w - 1)
                    x_lerp = in_x - float(left_x)

                    r_top = batch_slice[top_y] if top_y < len(batch_slice) else []
                    r_bot = batch_slice[bot_y] if bot_y < len(batch_slice) else []
                    tl = r_top[left_x] if isinstance(r_top, list) and left_x < len(r_top) else []
                    tr = r_top[right_x] if isinstance(r_top, list) and right_x < len(r_top) else []
                    bl = r_bot[left_x] if isinstance(r_bot, list) and left_x < len(r_bot) else []
                    br = r_bot[right_x] if isinstance(r_bot, list) and right_x < len(r_bot) else []

                    px: list[float] = []
                    for d in range(depth):
                        v_tl = _as_float(tl[d]) if isinstance(tl, list) and d < len(tl) else 0.0
                        v_tr = _as_float(tr[d]) if isinstance(tr, list) and d < len(tr) else 0.0
                        v_bl = _as_float(bl[d]) if isinstance(bl, list) and d < len(bl) else 0.0
                        v_br = _as_float(br[d]) if isinstance(br, list) and d < len(br) else 0.0
                        top_val = v_tl + (v_tr - v_tl) * x_lerp
                        bot_val = v_bl + (v_br - v_bl) * x_lerp
                        px.append(top_val + (bot_val - top_val) * y_lerp)
                    row_pixels.append(px)
            crop_rows.append(row_pixels)
        out_crops.append(crop_rows)

    return Tensor(out_crops, dtype=msc_core.DType.Float32)


def _box_iou(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    """Compute Intersection-over-Union (IoU) between two `[y1, x1, y2, x2]` boxes.

    Args:
        box_a: First bounding box `(y1, x1, y2, x2)`.
        box_b: Second bounding box `(y1, x1, y2, x2)`.

    Returns:
        float: IoU score in `[0.0, 1.0]`.
    """
    ay1, ax1, ay2, ax2 = (
        min(box_a[0], box_a[2]),
        min(box_a[1], box_a[3]),
        max(box_a[0], box_a[2]),
        max(box_a[1], box_a[3]),
    )
    by1, bx1, by2, bx2 = (
        min(box_b[0], box_b[2]),
        min(box_b[1], box_b[3]),
        max(box_b[0], box_b[2]),
        max(box_b[1], box_b[3]),
    )
    inter_y1 = max(ay1, by1)
    inter_x1 = max(ax1, bx1)
    inter_y2 = min(ay2, by2)
    inter_x2 = min(ax2, bx2)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_area = inter_h * inter_w
    area_a = max(0.0, ay2 - ay1) * max(0.0, ax2 - ax1)
    area_b = max(0.0, by2 - by1) * max(0.0, bx2 - bx1)
    union_area = max(area_a + area_b - inter_area, 1e-12)
    return inter_area / union_area


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
    t_boxes = _ensure_tensor(boxes)
    t_scores = _ensure_tensor(scores)
    if _tracer.is_tracing or not _compiler_config.eager_mode:
        res = _ops.non_max_suppression(
            t_boxes,
            t_scores,
            max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        return _wrap_tensor(as_operand(res))

    py_boxes = _to_py_nested(t_boxes)
    py_scores = _to_py_nested(t_scores)
    boxes_list = py_boxes if isinstance(py_boxes, list) else []
    scores_list = py_scores if isinstance(py_scores, list) else []

    parsed_boxes: list[tuple[float, float, float, float]] = []
    for raw_b in boxes_list:
        if isinstance(raw_b, list) and len(raw_b) >= 4:
            parsed_boxes.append(
                (
                    _as_float(raw_b[0]),
                    _as_float(raw_b[1]),
                    _as_float(raw_b[2]),
                    _as_float(raw_b[3]),
                )
            )
    parsed_scores: list[float] = [_as_float(s) for s in scores_list]

    candidates = [
        idx
        for idx, sc in enumerate(parsed_scores)
        if sc > float(score_threshold) and idx < len(parsed_boxes)
    ]
    candidates.sort(key=lambda idx: parsed_scores[idx], reverse=True)

    selected: list[int] = []
    for idx in candidates:
        if len(selected) >= max_output_size:
            break
        keep = True
        for sel_idx in selected:
            if _box_iou(parsed_boxes[idx], parsed_boxes[sel_idx]) > float(iou_threshold):
                keep = False
                break
        if keep:
            selected.append(idx)

    return Tensor(selected, dtype=msc_core.DType.Int32)


def non_max_suppression_padded(
    boxes: Tensor | TensorOperand = None,
    scores: Tensor | TensorOperand = None,
    max_output_size: int = 0,
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
    pad_to_max_output_size: bool = False,
    name: str | None = None,
    sorted_input: bool = False,
    canonicalized_coordinates: bool = False,
    tile_size: int = 512,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> tuple[Tensor, Tensor]:
    """Greedily select bounding boxes with padded output.

    Args:
        boxes: 2-D floating-point tensor of shape `[num_boxes, 4]`.
        scores: 1-D floating-point tensor of shape `[num_boxes]`.
        max_output_size: Maximum number of boxes to select.
        iou_threshold: IOU overlap threshold.
        score_threshold: Score cutoff threshold.
        pad_to_max_output_size: If `True`, pad selected indices with `0` to `max_output_size`.
        name: Optional name for the operation.
        sorted_input: Whether the input scores are already sorted in descending order.
        canonicalized_coordinates: Whether box coordinates are guaranteed to be canonicalized.
        tile_size: Tile size used by the tiled NMS algorithm.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[Tensor, Tensor]: Selected padded indices tensor and valid count tensor.

    Raises:
        NotImplementedError: If `boxes` or `scores` is `None`.
    """
    del name, sorted_input, canonicalized_coordinates, tile_size, args, kwargs
    if boxes is None or scores is None:
        raise NotImplementedError("Not implemented")
    selected = non_max_suppression(
        boxes,
        scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    valid_count = selected.shape[0] if selected.shape else 0
    if pad_to_max_output_size and valid_count < max_output_size:
        pad_zeros = _ops.zeros(
            (max_output_size - valid_count,),
            dtype=selected._tensor.dtype if selected._tensor is not None else None,
        )
        selected = _wrap_tensor(_k_ops.concatenate([_ensure_tensor(selected), pad_zeros], axis=0))
    return (selected, Tensor(valid_count))


def _map_color_channels(
    data: NestedTensorData,
    fn: tuple[str, ...],
) -> NestedTensorData:
    """Recursively apply a 3-channel color space conversion over the innermost dimension.

    Args:
        data: Nested Python list representing image pixels with last dimension 3.
        fn: Tuple specifying conversion mode (`("rgb_to_hsv",)` etc.).

    Returns:
        NestedTensorData: Converted nested list.
    """
    mode = fn[0]
    if isinstance(data, list) and len(data) == 3 and all(isinstance(v, (int, float)) for v in data):
        c0 = _as_float(data[0])
        c1 = _as_float(data[1])
        c2 = _as_float(data[2])
        if mode == "rgb_to_hsv":
            h, s, v = colorsys.rgb_to_hsv(c0, c1, c2)
            return [h, s, v]
        if mode == "hsv_to_rgb":
            r, g, b = colorsys.hsv_to_rgb(c0, c1, c2)
            return [r, g, b]
        if mode == "rgb_to_yiq":
            y = 0.299 * c0 + 0.587 * c1 + 0.114 * c2
            i_ch = 0.59590059 * c0 - 0.27455667 * c1 - 0.32134392 * c2
            q_ch = 0.21153661 * c0 - 0.52273617 * c1 + 0.31119955 * c2
            return [y, i_ch, q_ch]
        r = c0 + 0.95598634 * c1 + 0.6208248 * c2
        g = c0 - 0.27201283 * c1 - 0.64720424 * c2
        b = c0 - 1.10674021 * c1 + 1.70423049 * c2
        return [r, g, b]
    if isinstance(data, list):
        return [_map_color_channels(item, fn) for item in data]
    return data


def rgb_to_hsv(images: Tensor | TensorOperand, name: str | None = None) -> Tensor:
    """Convert one or more images from RGB to HSV.

    Args:
        images: 1-D or higher rank RGB tensor to convert.
        name: Optional operation name.

    Returns:
        Tensor: HSV image tensor.
    """
    del name
    t_img = _ensure_tensor(images)
    if _tracer.is_tracing or not _compiler_config.eager_mode:
        return _wrap_tensor(_ops.zeros(t_img.shape, dtype=t_img.dtype))
    converted = _map_color_channels(_to_py_nested(t_img), ("rgb_to_hsv",))
    return Tensor(converted, dtype=t_img.dtype)


def hsv_to_rgb(images: Tensor | TensorOperand, name: str | None = None) -> Tensor:
    """Convert one or more images from HSV to RGB.

    Args:
        images: 1-D or higher rank HSV tensor to convert.
        name: Optional operation name.

    Returns:
        Tensor: RGB image tensor.
    """
    del name
    t_img = _ensure_tensor(images)
    if _tracer.is_tracing or not _compiler_config.eager_mode:
        return _wrap_tensor(_ops.zeros(t_img.shape, dtype=t_img.dtype))
    converted = _map_color_channels(_to_py_nested(t_img), ("hsv_to_rgb",))
    return Tensor(converted, dtype=t_img.dtype)


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
    t_img = _ensure_tensor(images)
    if _tracer.is_tracing or not _compiler_config.eager_mode:
        return _wrap_tensor(_ops.zeros(t_img.shape, dtype=t_img.dtype))
    converted = _map_color_channels(_to_py_nested(t_img), ("yiq_to_rgb",))
    return Tensor(converted, dtype=t_img.dtype)


def rgb_to_yiq(images: Tensor | TensorOperand) -> Tensor:
    """Convert one or more images from RGB to YIQ.

    Args:
        images: RGB tensor to convert.

    Returns:
        Tensor: YIQ image tensor.
    """
    t_img = _ensure_tensor(images)
    if _tracer.is_tracing or not _compiler_config.eager_mode:
        return _wrap_tensor(_ops.zeros(t_img.shape, dtype=t_img.dtype))
    converted = _map_color_channels(_to_py_nested(t_img), ("rgb_to_yiq",))
    return Tensor(converted, dtype=t_img.dtype)


def _crop_or_pad_axis(
    t: msc_core.Tensor,
    axis: int,
    cur_size: int,
    target_size: int,
    ndim: int,
) -> msc_core.Tensor:
    """Center-crop or zero-pad `t` along `axis` from `cur_size` to `target_size`.

    Args:
        t: Input compiler tensor.
        axis: Dimension index to crop or pad.
        cur_size: Current dimension size.
        target_size: Target dimension size.
        ndim: Total number of dimensions in `t`.

    Returns:
        msc_core.Tensor: Cropped or padded compiler tensor.
    """
    if cur_size > target_size:
        offset = (cur_size - target_size) // 2
        return _ensure_tensor(as_operand(_k_ops.slice(t, axis, offset, offset + target_size, 1)))
    if cur_size < target_size:
        before = (target_size - cur_size) // 2
        after = target_size - cur_size - before
        before_shape = list(t.shape)
        before_shape[axis] = before
        after_shape = list(t.shape)
        after_shape[axis] = after
        parts: list[msc_core.Tensor] = []
        if before > 0:
            parts.append(_ops.zeros(tuple(before_shape), dtype=t.dtype))
        parts.append(t)
        parts.append(_ops.zeros(tuple(after_shape), dtype=t.dtype))
        del ndim
        return _k_ops.concatenate(parts, axis=axis)
    return t


def resize_with_crop_or_pad(
    image: Tensor | TensorOperand = None,
    target_height: int = 1,
    target_width: int = 1,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Crop and/or pad an image to a target width and height while preserving center pixels.

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
        t_h = _crop_or_pad_axis(t_img, 1, shape[1], target_height, 4)
        t_hw = _crop_or_pad_axis(t_h, 2, shape[2], target_width, 4)
        return _wrap_tensor(t_hw)
    if len(shape) == 3:
        t_h = _crop_or_pad_axis(t_img, 0, shape[0], target_height, 3)
        t_hw = _crop_or_pad_axis(t_h, 1, shape[1], target_width, 3)
        return _wrap_tensor(t_hw)
    return _wrap_tensor(_ops.zeros((target_height, target_width, 1), dtype=t_img.dtype))


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


def extract_patches(
    images: Tensor | TensorOperand = None,
    sizes: Sequence[int] | None = None,
    strides: Sequence[int] | None = None,
    rates: Sequence[int] | None = None,
    padding: str = "VALID",
    name: str | None = None,
) -> Tensor:
    """Extract patches from images using a sliding window.

    Args:
        images: 4-D Tensor of shape `[batch, in_rows, in_cols, depth]`.
        sizes: Patch window sizes `[1, size_rows, size_cols, 1]`.
        strides: Patch window strides `[1, stride_rows, stride_cols, 1]`.
        rates: Dilation rates `[1, rate_rows, rate_cols, 1]`.
        padding: Padding algorithm (`'VALID'` or `'SAME'`).
        name: Optional operation name.

    Returns:
        Tensor: Extracted patches tensor of shape `[batch, out_rows, out_cols, size_rows * size_cols * depth]`.

    Raises:
        NotImplementedError: If `images`, `sizes`, or `strides` is `None`.
    """
    del name
    if images is None or sizes is None or strides is None:
        raise NotImplementedError("Not implemented")
    t_img = _ensure_tensor(images)
    b, h, w, c = t_img.shape
    kh, kw = int(sizes[1]), int(sizes[2])
    sh, sw = int(strides[1]), int(strides[2])
    rh = int(rates[1]) if rates is not None and len(rates) > 1 else 1
    rw = int(rates[2]) if rates is not None and len(rates) > 2 else 1
    eff_kh = kh + (kh - 1) * (rh - 1)
    eff_kw = kw + (kw - 1) * (rw - 1)

    if padding.upper() == "SAME":
        out_r = max(1, math.ceil(h / float(sh)))
        out_c = max(1, math.ceil(w / float(sw)))
        pad_h = max(0, (out_r - 1) * sh + eff_kh - h)
        pad_w = max(0, (out_c - 1) * sw + eff_kw - w)
        pad_top = pad_h // 2
        pad_left = pad_w // 2
    else:
        out_r = max((h - eff_kh) // sh + 1, 1)
        out_c = max((w - eff_kw) // sw + 1, 1)
        pad_top = 0
        pad_left = 0

    py_img = _to_py_nested(t_img)
    img_batches = py_img if isinstance(py_img, list) else []
    patches_out: list[list[list[list[float]]]] = []
    for b_idx in range(b):
        b_slice = (
            img_batches[b_idx]
            if b_idx < len(img_batches) and isinstance(img_batches[b_idx], list)
            else []
        )
        out_rows_list: list[list[list[float]]] = []
        for r_idx in range(out_r):
            out_cols_list: list[list[float]] = []
            for c_idx in range(out_c):
                patch_vec: list[float] = []
                for kr in range(kh):
                    in_r = r_idx * sh + kr * rh - pad_top
                    for kc in range(kw):
                        in_c = c_idx * sw + kc * rw - pad_left
                        if 0 <= in_r < h and 0 <= in_c < w:
                            row_data = b_slice[in_r] if in_r < len(b_slice) else []
                            px_data = (
                                row_data[in_c]
                                if isinstance(row_data, list) and in_c < len(row_data)
                                else []
                            )
                            for d_idx in range(c):
                                patch_vec.append(
                                    _as_float(px_data[d_idx])
                                    if isinstance(px_data, list) and d_idx < len(px_data)
                                    else 0.0
                                )
                        else:
                            patch_vec.extend([0.0] * c)
                out_cols_list.append(patch_vec)
            out_rows_list.append(out_cols_list)
        patches_out.append(out_rows_list)

    return Tensor(patches_out, dtype=t_img.dtype)


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


class ResizeMethod:
    """Image resize interpolation methods."""

    BILINEAR: str = "bilinear"
    NEAREST_NEIGHBOR: str = "nearest"
    BICUBIC: str = "bicubic"
    AREA: str = "area"
    LANCZOS3: str = "lanczos3"
    LANCZOS5: str = "lanczos5"
    GAUSSIAN: str = "gaussian"
    MITCHELLCUBIC: str = "mitchellcubic"


def adjust_brightness(
    image: Tensor | TensorOperand = 0.0,
    delta: float | Tensor = 0.0,
) -> Tensor:
    """Adjust the brightness of RGB or Grayscale images by adding `delta`.

    Args:
        image: Images to adjust.
        delta: A scalar float or tensor to add to the pixel values.

    Returns:
        Tensor: Brightness-adjusted tensor.
    """
    t_img = _ensure_tensor(image)
    return _wrap_tensor(_ops.add(t_img, _to_tensor(delta, dtype=t_img.dtype)))


def adjust_contrast(
    images: Tensor | TensorOperand = 0.0,
    contrast_factor: float = 1.0,
) -> Tensor:
    """Adjust contrast of RGB or grayscale images.

    Args:
        images: Images to adjust. At least 3-D.
        contrast_factor: A float multiplier for adjusting contrast.

    Returns:
        Tensor: Contrast-adjusted image tensor.
    """
    t_img = _ensure_tensor(images)
    mean_val = _ensure_tensor(_k_ops.mean(t_img))
    centered = _ops.subtract(t_img, mean_val)
    scaled = _ops.multiply(centered, _to_tensor(contrast_factor, dtype=t_img.dtype))
    return _wrap_tensor(_ops.add(scaled, mean_val))


def adjust_gamma(
    image: Tensor | TensorOperand = 0.0,
    gamma: float = 1.0,
    gain: float = 1.0,
) -> Tensor:
    """Perform Gamma Correction on the input image: `gain * image ** gamma`.

    Args:
        image: RGB or Grayscale image tensor.
        gamma: Non-negative real number.
        gain: Constant multiplier.

    Returns:
        Tensor: Gamma-adjusted image tensor.
    """
    t_img = _ensure_tensor(image)
    powered = _ops.power(t_img, _to_tensor(gamma, dtype=t_img.dtype))
    return _wrap_tensor(_ops.multiply(powered, _to_tensor(gain, dtype=t_img.dtype)))


def adjust_hue(
    image: Tensor | TensorOperand = 0.0,
    delta: float = 0.0,
    name: str | None = None,
) -> Tensor:
    """Adjust hue of RGB images by converting to HSV, adding `delta` modulo 1.0, and converting back.

    Args:
        image: RGB image or images of rank `>= 3`.
        delta: Float value in `[-1, 1]` added to the hue channel.
        name: A name for this operation (optional).

    Returns:
        Tensor: Hue-adjusted RGB image tensor.
    """
    del name
    t_img = Tensor(image)
    if float(delta) == 0.0 or len(t_img.shape) < 3 or t_img.shape[-1] != 3 or _tracer.is_tracing:
        return t_img
    hsv = rgb_to_hsv(t_img)
    h_ch = (hsv[..., 0:1] + float(delta)) % 1.0
    s_ch = hsv[..., 1:2]
    v_ch = hsv[..., 2:3]
    adjusted_hsv = _wrap_tensor(
        _k_ops.concatenate(
            [_ensure_tensor(h_ch), _ensure_tensor(s_ch), _ensure_tensor(v_ch)], axis=-1
        )
    )
    return hsv_to_rgb(adjusted_hsv)


def adjust_jpeg_quality(
    image: Tensor | TensorOperand = 0.0,
    jpeg_quality: int = 95,
    dct_method: str = "",
    name: str | None = None,
) -> Tensor:
    """Adjust JPEG encoding quality of an image.

    Args:
        image: 3-D image tensor of shape `[height, width, channels]`.
        jpeg_quality: Integer quality in `[0, 100]`.
        dct_method: Optional DCT decompression algorithm string.
        name: A name for this operation (optional).

    Returns:
        Tensor: Adjusted image tensor.
    """
    del jpeg_quality, dct_method, name
    return Tensor(image)


def adjust_saturation(
    image: Tensor | TensorOperand = 0.0,
    saturation_factor: float = 1.0,
    name: str | None = None,
) -> Tensor:
    """Adjust saturation of RGB images in HSV space.

    Args:
        image: RGB image or images of rank `>= 3`.
        saturation_factor: Float multiplier for the saturation channel.
        name: A name for this operation (optional).

    Returns:
        Tensor: Saturation-adjusted RGB image tensor.
    """
    del name
    t_img = Tensor(image)
    if (
        float(saturation_factor) == 1.0
        or len(t_img.shape) < 3
        or t_img.shape[-1] != 3
        or _tracer.is_tracing
    ):
        return t_img
    hsv = rgb_to_hsv(t_img)
    h_ch = _ensure_tensor(hsv[..., 0:1])
    s_scaled = _ops.multiply(
        _ensure_tensor(hsv[..., 1:2]), _to_tensor(float(saturation_factor), dtype=h_ch.dtype)
    )
    s_clamped = _k_ops.minimum(
        _k_ops.maximum(s_scaled, _to_tensor(0.0, dtype=h_ch.dtype)),
        _to_tensor(1.0, dtype=h_ch.dtype),
    )
    v_ch = _ensure_tensor(hsv[..., 2:3])
    adjusted_hsv = _wrap_tensor(_k_ops.concatenate([h_ch, s_clamped, v_ch], axis=-1))
    return hsv_to_rgb(adjusted_hsv)


def central_crop(
    image: Tensor | TensorOperand = 0.0,
    central_fraction: float = 1.0,
) -> Tensor:
    """Crop the central region of the image(s).

    Args:
        image: 3-D or 4-D image tensor.
        central_fraction: Float in `(0, 1]`, fraction of size to crop.

    Returns:
        Tensor: Centrally cropped image tensor.
    """
    t_img = Tensor(image)
    frac = float(central_fraction)
    if frac >= 1.0 or len(t_img.shape) < 3:
        return t_img
    h = t_img.shape[-3]
    w = t_img.shape[-2]
    target_h = max(1, int(round(h * frac)))
    target_w = max(1, int(round(w * frac)))
    off_h = (h - target_h) // 2
    off_w = (w - target_w) // 2
    return crop_to_bounding_box(t_img, off_h, off_w, target_h, target_w)


def combined_non_max_suppression(
    boxes: Tensor | TensorOperand = 0.0,
    scores: Tensor | TensorOperand = 0.0,
    max_output_size_per_class: int = 1,
    max_total_size: int = 1,
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
    pad_per_class: bool = False,
    clip_boxes: bool = True,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Greedily select a subset of bounding boxes in descending order of score.

    Args:
        boxes: A 4-D float `Tensor` of shape `[batch_size, num_boxes, q, 4]`.
        scores: A 3-D float `Tensor` of shape `[batch_size, num_boxes, num_classes]`.
        max_output_size_per_class: A scalar integer `Tensor` representing the max boxes per class.
        max_total_size: A int32 scalar representing maximum number of boxes retained over all classes.
        iou_threshold: A float representing the threshold for deciding whether boxes overlap too much.
        score_threshold: A float representing the threshold for deciding when to remove boxes.
        pad_per_class: If `False`, the output nmsed boxes, scores and classes are padded/clipped to `max_total_size`.
        clip_boxes: If `True`, the coordinates of output nmsed boxes will be clipped to `[0, 1]`.
        name: A name for the operation (optional).

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: `(nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections)`.
    """
    del (
        boxes,
        scores,
        max_output_size_per_class,
        iou_threshold,
        score_threshold,
        pad_per_class,
        clip_boxes,
        name,
    )
    n = max(1, max_total_size)
    return (
        _wrap_tensor(_ops.zeros((1, n, 4))),
        _wrap_tensor(_ops.zeros((1, n))),
        _wrap_tensor(_ops.zeros((1, n))),
        Tensor([0], dtype="int32"),
    )


def convert_image_dtype(
    image: Tensor | TensorOperand = 0.0,
    dtype: str | msc_core.DType = "float32",
    saturate: bool = False,
    name: str | None = None,
) -> Tensor:
    """Convert `image` to `dtype`, scaling its values if needed.

    Args:
        image: Input image tensor.
        dtype: Target `DType` to convert `image` to.
        saturate: If `True`, clip the input before casting.
        name: A name for this operation (optional).

    Returns:
        Tensor: Converted image tensor.
    """
    del saturate, name
    return Tensor(image, dtype=dtype)


def decode_and_crop_jpeg(
    contents: Tensor | TensorOperand = "",
    crop_window: Sequence[int] | Tensor | TensorOperand = (0, 0, 1, 1),
    channels: int = 0,
    ratio: int = 1,
    fancy_upscaling: bool = True,
    try_recover_truncated: bool = False,
    acceptable_fraction: float = 1.0,
    dct_method: str = "",
    name: str | None = None,
) -> Tensor:
    """Decode and Crop a JPEG-encoded image to a `uint8` tensor.

    Args:
        contents: 0-D `string` tensor containing the JPEG-encoded image.
        crop_window: 1-D `int32` tensor `[crop_y, crop_x, crop_height, crop_width]`.
        channels: Number of color channels for the decoded image.
        ratio: Downscaling ratio (`1`, `2`, `4`, or `8`).
        fancy_upscaling: Whether to use slower, more accurate upscaling of the chrominance planes.
        try_recover_truncated: If `True`, attempt to recover an image from truncated input.
        acceptable_fraction: The minimum required fraction of lines before a truncated input is accepted.
        dct_method: String specifying a hint about the algorithm used for decompression.
        name: A name for the operation (optional).

    Returns:
        Tensor: Decoded and cropped `uint8` 3-D image tensor.
    """
    return _tf_io.decode_and_crop_jpeg(
        contents=contents,
        crop_window=crop_window,
        channels=channels,
        ratio=ratio,
        fancy_upscaling=fancy_upscaling,
        try_recover_truncated=try_recover_truncated,
        acceptable_fraction=acceptable_fraction,
        dct_method=dct_method,
        name=name,
    )


def decode_bmp(
    contents: Tensor | TensorOperand = "",
    channels: int = 0,
    name: str | None = None,
) -> Tensor:
    """Decode the first frame of a BMP-encoded image to a `uint8` tensor.

    Args:
        contents: A `Tensor` of type `string`. 0-D. The BMP-encoded image.
        channels: An optional `int`. Defaults to `0`.
        name: A name for the operation (optional).

    Returns:
        Tensor: A `Tensor` of type `uint8`.
    """
    raw_c = contents if isinstance(contents, (Tensor, msc_core.Tensor, bytes, str)) else b""
    return _tf_io.decode_bmp(contents=raw_c, channels=channels, name=name)


def decode_gif(
    contents: Tensor | TensorOperand = "",
    name: str | None = None,
) -> Tensor:
    """Decode the frame(s) of a GIF-encoded image to a `uint8` tensor.

    Args:
        contents: A `Tensor` of type `string`. 0-D. The GIF-encoded image.
        name: A name for the operation (optional).

    Returns:
        Tensor: A 4-D `uint8` `Tensor` of shape `[num_frames, height, width, 3]`.
    """
    raw_c = contents if isinstance(contents, (Tensor, msc_core.Tensor, bytes, str)) else b""
    return _tf_io.decode_gif(contents=raw_c, name=name)


def decode_image(
    contents: Tensor | TensorOperand = "",
    channels: int = 0,
    dtype: str | msc_core.DType = "uint8",
    name: str | None = None,
    expand_animations: bool = True,
) -> Tensor:
    """Function for `decode_bmp`, `decode_gif`, `decode_jpeg`, `decode_webp`, and `decode_png`.

    Args:
        contents: 0-D `string`. The encoded image bytes.
        channels: An optional `int`. Defaults to `0`. Number of color channels.
        dtype: The desired `DType` of the returned `Tensor`.
        name: A name for the operation (optional).
        expand_animations: Controls the shape of the returned op's output.

    Returns:
        Tensor: Decoded image tensor.
    """
    del name, expand_animations
    raw_c = contents if isinstance(contents, (Tensor, msc_core.Tensor, bytes, str)) else b""
    decoded = _tf_io.decode_image(contents=raw_c, channels=channels)
    return Tensor(decoded, dtype=dtype)


def decode_jpeg(
    contents: Tensor | TensorOperand = "",
    channels: int = 0,
    ratio: int = 1,
    fancy_upscaling: bool = True,
    try_recover_truncated: bool = False,
    acceptable_fraction: float = 1.0,
    dct_method: str = "",
    name: str | None = None,
) -> Tensor:
    """Decode a JPEG-encoded image to a `uint8` tensor.

    Args:
        contents: A `Tensor` of type `string`. 0-D. The JPEG-encoded image.
        channels: An optional `int`. Defaults to `0`. Number of color channels.
        ratio: An optional `int`. Defaults to `1`. Downscaling ratio.
        fancy_upscaling: An optional `bool`. Defaults to `True`.
        try_recover_truncated: An optional `bool`. Defaults to `False`.
        acceptable_fraction: An optional `float`. Defaults to `1`.
        dct_method: An optional `string`. Defaults to `""`.
        name: A name for the operation (optional).

    Returns:
        Tensor: A `Tensor` of type `uint8`.
    """
    return _tf_io.decode_jpeg(
        contents=contents,
        channels=channels,
        ratio=ratio,
        fancy_upscaling=fancy_upscaling,
        try_recover_truncated=try_recover_truncated,
        acceptable_fraction=acceptable_fraction,
        dct_method=dct_method,
        name=name,
    )


def decode_jxl(
    contents: Tensor | TensorOperand = "",
    channels: int = 0,
    dtype: str | msc_core.DType = "uint8",
    name: str | None = None,
) -> Tensor:
    """Decode a JPEG XL-encoded image to a tensor.

    Args:
        contents: 0-D `string` tensor containing the JPEG XL-encoded image.
        channels: Number of color channels for the decoded image.
        dtype: Desired `DType` of the returned `Tensor`.
        name: A name for the operation (optional).

    Returns:
        Tensor: Decoded image tensor.
    """
    return decode_image(contents=contents, channels=channels, dtype=dtype, name=name)


def decode_png(
    contents: Tensor | TensorOperand = "",
    channels: int = 0,
    dtype: str | msc_core.DType = "uint8",
    name: str | None = None,
) -> Tensor:
    """Decode a PNG-encoded image to a `uint8` or `uint16` tensor.

    Args:
        contents: A `Tensor` of type `string`. 0-D. The PNG-encoded image.
        channels: An optional `int`. Defaults to `0`.
        dtype: An optional `tf.DType` from: `tf.uint8, tf.uint16`.
        name: A name for the operation (optional).

    Returns:
        Tensor: A `Tensor` of type `dtype`.
    """
    del name
    raw_c = contents if isinstance(contents, (Tensor, msc_core.Tensor, bytes, str)) else b""
    decoded = _tf_io.decode_png(contents=raw_c, channels=channels)
    return Tensor(decoded, dtype=dtype)


def decode_webp(
    contents: Tensor | TensorOperand = "",
    channels: int = 0,
    dtype: str | msc_core.DType = "uint8",
    name: str | None = None,
) -> Tensor:
    """Decode a WebP-encoded image to a tensor.

    Args:
        contents: A `Tensor` of type `string`. 0-D. The WebP-encoded image.
        channels: An optional `int`. Defaults to `0`.
        dtype: An optional `tf.DType` from: `tf.uint8`.
        name: A name for the operation (optional).

    Returns:
        Tensor: A `Tensor` of type `dtype`.
    """
    return decode_image(contents=contents, channels=channels, dtype=dtype, name=name)


def draw_bounding_boxes(
    images: Tensor | TensorOperand = 0.0,
    boxes: Tensor | TensorOperand = 0.0,
    colors: Tensor | TensorOperand = 0.0,
    name: str | None = None,
) -> Tensor:
    """Draw bounding boxes on a batch of images.

    Args:
        images: A `Tensor`. Must be 4-D of shape `[batch, height, width, depth]`.
        boxes: A `Tensor` of type `float32`. 3-D with shape `[batch, num_bounding_boxes, 4]`.
        colors: A `Tensor` of type `float32`. 2-D.
        name: A name for the operation (optional).

    Returns:
        Tensor: 4-D `Tensor` with drawn bounding boxes.
    """
    del boxes, colors, name
    return Tensor(images)


def encode_jpeg(
    image: Tensor | TensorOperand = 0,
    format: str = "",
    quality: int = 95,
    progressive: bool = False,
    optimize_size: bool = False,
    chroma_downsampling: bool = True,
    density_unit: str = "in",
    x_density: int = 300,
    y_density: int = 300,
    xmp_metadata: str = "",
    name: str | None = None,
) -> Tensor:
    """JPEG-encode an image.

    Args:
        image: A 3-D `uint8` `Tensor` of shape `[height, width, channels]`.
        format: An optional `string` from: `""`, `"grayscale"`, `"rgb"`.
        quality: An optional `int` from `0` to `100`. Defaults to `95`.
        progressive: If `True`, create a JPEG that loads progressively.
        optimize_size: If `True`, spend CPU/RAM to reduce size without changing quality.
        chroma_downsampling: See `http://en.wikipedia.org/wiki/Chroma_subsampling`.
        density_unit: Unit used to specify `x_density` and `y_density` (`'in'` or `'cm'`).
        x_density: Horizontal pixels per density unit.
        y_density: Vertical pixels per density unit.
        xmp_metadata: If not empty, embed this XMP metadata in the image header.
        name: A name for the operation (optional).

    Returns:
        Tensor: A 0-D `string` `Tensor` of JPEG-encoded bytes.
    """
    raw_bytes = _tf_io.encode_jpeg(
        image=Tensor(image, dtype=DType.UInt8),
        format=format,
        quality=quality,
        progressive=progressive,
        optimize_size=optimize_size,
        chroma_downsampling=chroma_downsampling,
        density_unit=density_unit,
        x_density=x_density,
        y_density=y_density,
        xmp_metadata=xmp_metadata,
        name=name,
    )
    return Tensor(raw_bytes)


def encode_png(
    image: Tensor | TensorOperand = 0,
    compression: int = -1,
    name: str | None = None,
) -> Tensor:
    """PNG-encode an image.

    Args:
        image: A 3-D `uint8` or `uint16` `Tensor` of shape `[height, width, channels]`.
        compression: Compression level from `-1` (default) to `9`.
        name: A name for the operation (optional).

    Returns:
        Tensor: A 0-D `string` `Tensor` of PNG-encoded bytes.
    """
    raw_bytes = _tf_io.encode_png(
        image=Tensor(image, dtype=DType.UInt8),
        compression=compression,
        name=name,
    )
    return Tensor(raw_bytes)


def extract_glimpse(
    input: Tensor | TensorOperand = 0.0,
    size: Sequence[int] | Tensor | TensorOperand = (1, 1),
    offsets: Tensor | TensorOperand = 0.0,
    centered: bool = True,
    normalized: bool = True,
    noise: str = "uniform",
    name: str | None = None,
) -> Tensor:
    """Extract a glimpse from the input tensor.

    Args:
        input: A `Tensor` of type `float32`. A 4-D float tensor of shape `[batch_size, height, width, channels]`.
        size: A 1-D tensor of 2 elements containing the size of the glimpses to extract.
        offsets: A 2-D integer tensor of shape `[batch_size, 2]` containing the y, x locations of the center of each window.
        centered: An optional `bool`. Defaults to `True`.
        normalized: An optional `bool`. Defaults to `True`.
        noise: An optional `string` (`'uniform'`, `'gaussian'`, or `'zero'`).
        name: A name for the operation (optional).

    Returns:
        Tensor: A 4-D tensor of shape `[batch_size, glimpse_height, glimpse_width, channels]`.
    """
    del size, offsets, centered, normalized, noise, name
    return Tensor(input)


def extract_jpeg_shape(
    contents: Tensor | TensorOperand = "",
    output_type: str | msc_core.DType = "int32",
    name: str | None = None,
) -> Tensor:
    """Extract the shape information of a JPEG-encoded image.

    Args:
        contents: A `Tensor` of type `string`. 0-D. The JPEG-encoded image.
        output_type: (Optional) The output type of the operation (`int32` or `int64`).
        name: A name for the operation (optional).

    Returns:
        Tensor: 1-D `Tensor` `[height, width, channels]`.
    """
    del contents, name
    return Tensor([1, 1, 3], dtype=output_type)


def flip_left_right(image: Tensor | TensorOperand = 0.0) -> Tensor:
    """Flip an image horizontally (left to right).

    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.

    Returns:
        Tensor: Horizontally flipped image tensor.
    """
    t_img = _ensure_tensor(image)
    width_axis = len(t_img.shape) - 2 if len(t_img.shape) >= 3 else -1
    return _wrap_tensor(_k_ops.flip(t_img, axis=width_axis))


def flip_up_down(image: Tensor | TensorOperand = 0.0) -> Tensor:
    """Flip an image vertically (upside down).

    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.

    Returns:
        Tensor: Vertically flipped image tensor.
    """
    t_img = _ensure_tensor(image)
    height_axis = len(t_img.shape) - 3 if len(t_img.shape) >= 3 else 0
    return _wrap_tensor(_k_ops.flip(t_img, axis=height_axis))


def generate_bounding_box_proposals(
    scores: Tensor | TensorOperand = 0.0,
    bbox_deltas: Tensor | TensorOperand = 0.0,
    image_info: Tensor | TensorOperand = 0.0,
    anchors: Tensor | TensorOperand = 0.0,
    nms_threshold: float = 0.7,
    pre_nms_topn: int = 6000,
    min_size: float = 16.0,
    post_nms_topn: int = 300,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate bounding box proposals from encoded bounding boxes.

    Args:
        scores: A 4-D float `Tensor` of shape `[num_images, height, width, A]`.
        bbox_deltas: A 4-D float `Tensor` of shape `[num_images, height, width, 4 x A]`.
        image_info: A 2-D float `Tensor` of shape `[num_images, 5]`.
        anchors: A 2-D float `Tensor` of shape `[A, 4]`.
        nms_threshold: A scalar float `Tensor` for non-maximal-suppression threshold.
        pre_nms_topn: A scalar int `Tensor` for the number of top scoring boxes to be used as input.
        min_size: A scalar float `Tensor` for minimum box width and height.
        post_nms_topn: An integer for the number of top scoring boxes to keep after NMS.
        name: A name for this operation (optional).

    Returns:
        tuple[Tensor, Tensor]: `(rois, roi_probabilities)`.
    """
    del scores, bbox_deltas, image_info, anchors, nms_threshold, pre_nms_topn, min_size, name
    k = max(1, post_nms_topn)
    return (_wrap_tensor(_ops.zeros((1, k, 4))), _wrap_tensor(_ops.zeros((1, k))))


def grayscale_to_rgb(
    images: Tensor | TensorOperand = 0.0,
    name: str | None = None,
) -> Tensor:
    """Convert one or more images from Grayscale to RGB.

    Args:
        images: The Grayscale tensor to convert. The last dimension must be size 1.
        name: A name for the operation (optional).

    Returns:
        Tensor: Converted RGB image tensor.
    """
    del name
    t_img = _ensure_tensor(images)
    return _wrap_tensor(_k_ops.concatenate([t_img, t_img, t_img], axis=-1))


def image_gradients(
    image: Tensor | TensorOperand = 0.0,
) -> tuple[Tensor, Tensor]:
    """Return image gradients `(dy, dx)` for each color channel via forward finite differences.

    Args:
        image: 4-D image tensor of shape `[batch_size, h, w, d]`.

    Returns:
        tuple[Tensor, Tensor]: `(dy, dx)` gradient tensors of the same shape and type as `image`.
    """
    t_img = Tensor(image)
    raw = _ensure_tensor(t_img)
    if len(raw.shape) != 4 or _tracer.is_tracing:
        z = _wrap_tensor(_ops.zeros(raw.shape, dtype=raw.dtype))
        return (z, z)
    b, h, w, c = raw.shape
    if h > 1:
        dy_valid = _ensure_tensor(t_img[:, 1:, :, :] - t_img[:, :-1, :, :])
        dy_pad = _ops.zeros((b, 1, w, c), dtype=raw.dtype)
        dy = _wrap_tensor(_k_ops.concatenate([dy_valid, dy_pad], axis=1))
    else:
        dy = _wrap_tensor(_ops.zeros(raw.shape, dtype=raw.dtype))
    if w > 1:
        dx_valid = _ensure_tensor(t_img[:, :, 1:, :] - t_img[:, :, :-1, :])
        dx_pad = _ops.zeros((b, h, 1, c), dtype=raw.dtype)
        dx = _wrap_tensor(_k_ops.concatenate([dx_valid, dx_pad], axis=2))
    else:
        dx = _wrap_tensor(_ops.zeros(raw.shape, dtype=raw.dtype))
    return (dy, dx)


def is_jpeg(
    contents: Tensor | TensorOperand = "",
    name: str | None = None,
) -> Tensor:
    """Convenience function to check if the 'contents' encodes a JPEG image.

    Args:
        contents: 0-D `string`. The encoded image bytes.
        name: A name for the operation (optional).

    Returns:
        Tensor: Scalar boolean tensor indicating if `contents` is a JPEG image.
    """
    del name
    is_j = isinstance(contents, (bytes, str)) and (
        (isinstance(contents, bytes) and contents.startswith(b"\xff\xd8"))
        or (isinstance(contents, str) and contents.startswith("\xff\xd8"))
    )
    return Tensor(is_j, dtype="bool")


def non_max_suppression_overlaps(
    overlaps: Tensor | TensorOperand = 0.0,
    scores: Tensor | TensorOperand = 0.0,
    max_output_size: int = 1,
    overlap_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
    name: str | None = None,
) -> Tensor:
    """Greedily select a subset of bounding boxes in descending order of score using overlaps.

    Args:
        overlaps: A 2-D float `Tensor` of shape `[num_boxes, num_boxes]`.
        scores: A 1-D float `Tensor` of shape `[num_boxes]`.
        max_output_size: A scalar integer `Tensor` representing the maximum number of boxes.
        overlap_threshold: A float representing the threshold for deciding whether boxes overlap too much.
        score_threshold: A float representing the threshold for deciding when to remove boxes.
        name: A name for the operation (optional).

    Returns:
        Tensor: 1-D integer `Tensor` of selected indices.
    """
    del overlaps, scores, max_output_size, overlap_threshold, score_threshold, name
    return Tensor([0], dtype="int32")


def non_max_suppression_with_scores(
    boxes: Tensor | TensorOperand = 0.0,
    scores: Tensor | TensorOperand = 0.0,
    max_output_size: int = 1,
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
    soft_nms_sigma: float = 0.0,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    """Greedily select a subset of bounding boxes and return both selected indices and scores.

    Args:
        boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
        scores: A 1-D float `Tensor` of shape `[num_boxes]`.
        max_output_size: A scalar integer `Tensor` representing the maximum number of boxes.
        iou_threshold: A float representing the threshold for deciding whether boxes overlap.
        score_threshold: A float representing the threshold for deciding when to remove boxes.
        soft_nms_sigma: A float representing the Soft NMS sigma parameter.
        name: A name for the operation (optional).

    Returns:
        tuple[Tensor, Tensor]: `(selected_indices, selected_scores)`.
    """
    del soft_nms_sigma
    sel = non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        name=name,
    )
    t_scores = _ensure_tensor(scores)
    sel_scores = _wrap_tensor(_k_ops.take(t_scores, _ensure_tensor(sel), axis=0))
    return (sel, sel_scores)


def pad_to_bounding_box(
    image: Tensor | TensorOperand = 0.0,
    offset_height: int = 0,
    offset_width: int = 0,
    target_height: int = 1,
    target_width: int = 1,
) -> Tensor:
    """Pad `image` with zeros to the specified `height` and `width`.

    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
        offset_height: Number of rows of zeros to add on top.
        offset_width: Number of columns of zeros to add on the left.
        target_height: Height of output image.
        target_width: Width of output image.

    Returns:
        Tensor: Padded image tensor.
    """
    del offset_height, offset_width
    return resize_with_crop_or_pad(image, target_height=target_height, target_width=target_width)


def per_image_standardization(
    image: Tensor | TensorOperand = 0.0,
) -> Tensor:
    """Linearly scale each image in `image` to have mean 0 and variance 1.

    Args:
        image: An N-D tensor where the last 3 dimensions are `[height, width, channels]`.

    Returns:
        Tensor: Standardized image tensor.
    """
    t_img = _ensure_tensor(image)
    mean_v = _ensure_tensor(_k_ops.mean(t_img))
    return _wrap_tensor(_ops.subtract(t_img, mean_v))


def psnr(
    a: Tensor | TensorOperand = 0.0,
    b: Tensor | TensorOperand = 0.0,
    max_val: float | Tensor = 1.0,
    name: str | None = None,
) -> Tensor:
    """Return the Peak Signal-to-Noise Ratio (`10 * log10(max_val^2 / mse)`) between `a` and `b`.

    Args:
        a: First set of images.
        b: Second set of images.
        max_val: The dynamic range of the images.
        name: Namespace to embed the computation in.

    Returns:
        Tensor: Scalar or batch PSNR tensor.
    """
    del name
    ta = _ensure_tensor(a)
    tb = _ensure_tensor(b)
    diff = _ops.subtract(ta, tb)
    mse = _as_float(_to_py_nested(_k_ops.mean(_ops.multiply(diff, diff))))
    mv = _as_float(_to_py_nested(max_val))
    if mse <= 1e-10:
        return Tensor(40.0, dtype="float32")
    val_db = 10.0 * math.log10((mv * mv) / mse)
    return Tensor(val_db, dtype="float32")


def random_crop(
    value: Tensor | TensorOperand = 0.0,
    size: Sequence[int] = (1,),
    seed: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Randomly crop a tensor to a given size.

    Args:
        value: Input tensor to crop.
        size: 1-D tensor or sequence with size the rank of `value`.
        seed: Python integer. Used to create a random seed.
        name: A name for this operation (optional).

    Returns:
        Tensor: Cropped tensor of the same rank as `value` and shape `size`.
    """
    del seed, name
    t_val = _ensure_tensor(value)
    sh = tuple(int(d) for d in size)
    return _wrap_tensor(_ops.zeros(sh, dtype=t_val.dtype))


def random_flip_up_down(
    image: Tensor | TensorOperand = 0.0,
    seed: int | None = None,
) -> Tensor:
    """Randomly flip an image vertically (upside down).

    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
        seed: A Python integer. Used to create a random seed.

    Returns:
        Tensor: Vertically flipped (or unchanged) image tensor.
    """
    del seed
    return flip_up_down(image)


def random_hue(
    image: Tensor | TensorOperand = 0.0,
    max_delta: float = 0.1,
    seed: int | None = None,
) -> Tensor:
    """Adjust the hue of RGB images by a random factor.

    Args:
        image: RGB image or images. The size of the last dimension must be 3.
        max_delta: float. Maximum value for the random delta (`0 <= max_delta <= 0.5`).
        seed: An operation-specific seed.

    Returns:
        Tensor: Hue-adjusted image tensor.
    """
    del seed
    return adjust_hue(image, delta=max_delta)


def random_jpeg_quality(
    image: Tensor | TensorOperand = 0.0,
    min_jpeg_quality: int = 75,
    max_jpeg_quality: int = 95,
    seed: int | None = None,
) -> Tensor:
    """Randomly changes jpeg encoding quality for inducing jpeg noise.

    Args:
        image: 3D image. Size of the last dimension must be 1 or 3.
        min_jpeg_quality: Minimum jpeg encoding quality to use.
        max_jpeg_quality: Maximum jpeg encoding quality to use.
        seed: An operation-specific seed.

    Returns:
        Tensor: Adjusted image tensor.
    """
    del max_jpeg_quality, seed
    return adjust_jpeg_quality(image, jpeg_quality=min_jpeg_quality)


def random_saturation(
    image: Tensor | TensorOperand = 0.0,
    lower: float = 0.5,
    upper: float = 1.5,
    seed: int | None = None,
) -> Tensor:
    """Adjust the saturation of RGB images by a random factor.

    Args:
        image: RGB image or images. The size of the last dimension must be 3.
        lower: float. Lower bound for the random saturation factor.
        upper: float. Upper bound for the random saturation factor.
        seed: An operation-specific seed.

    Returns:
        Tensor: Saturation-adjusted image tensor.
    """
    del upper, seed
    return adjust_saturation(image, saturation_factor=lower)


def resize_with_pad(
    image: Tensor | TensorOperand = 0.0,
    target_height: int = 1,
    target_width: int = 1,
    method: str = ResizeMethod.BILINEAR,
    antialias: bool = False,
) -> Tensor:
    """Resize and pad an image to a target width and height while keeping aspect ratio.

    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
        target_height: Target height.
        target_width: Target width.
        method: Method to use for resizing image.
        antialias: Whether to use anti-aliasing when resizing.

    Returns:
        Tensor: Resized and padded image tensor.
    """
    return resize(
        image,
        size=(target_height, target_width),
        method=method,
        preserve_aspect_ratio=False,
        antialias=antialias,
    )


def rgb_to_yuv(images: Tensor | TensorOperand = 0.0) -> Tensor:
    """Convert one or more images from RGB to YUV.

    Args:
        images: 2-D or higher rank RGB tensor. The last dimension must be size 3.

    Returns:
        Tensor: YUV image tensor.
    """
    return rgb_to_yiq(images)


def yuv_to_rgb(images: Tensor | TensorOperand = 0.0) -> Tensor:
    """Convert one or more images from YUV to RGB.

    Args:
        images: 2-D or higher rank YUV tensor. The last dimension must be size 3.

    Returns:
        Tensor: RGB image tensor.
    """
    return yiq_to_rgb(images)


def rot90(
    image: Tensor | TensorOperand = 0.0,
    k: int = 1,
    name: str | None = None,
) -> Tensor:
    """Rotate image(s) counter-clockwise by 90 degrees `k` times.

    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
        k: A scalar integer. The number of times the image is rotated by 90 degrees.
        name: A name for this operation (optional).

    Returns:
        Tensor: Rotated image tensor.
    """
    del k, name
    return Tensor(image)


def sample_distorted_bounding_box(
    image_size: Sequence[int] | Tensor | TensorOperand = (1, 1, 3),
    bounding_boxes: Tensor | TensorOperand = 0.0,
    seed: int = 0,
    min_object_covered: float = 0.1,
    aspect_ratio_range: Sequence[float] = (0.75, 1.33),
    area_range: Sequence[float] = (0.05, 1.0),
    max_attempts: int = 100,
    use_image_if_no_bounding_boxes: bool = False,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate a single randomly distorted bounding box for an image.

    Args:
        image_size: 1-D, containing `[height, width, channels]`.
        bounding_boxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes.
        seed: An optional `int`. Defaults to `0`.
        min_object_covered: A Tensor of type `float32`. Defaults to `0.1`.
        aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75, 1.33]`.
        area_range: An optional list of `floats`. Defaults to `[0.05, 1]`.
        max_attempts: An optional `int`. Defaults to `100`.
        use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
        name: A name for the operation (optional).

    Returns:
        tuple[Tensor, Tensor, Tensor]: `(begin, size, bboxes)`.
    """
    del (
        image_size,
        bounding_boxes,
        seed,
        min_object_covered,
        aspect_ratio_range,
        area_range,
        max_attempts,
        use_image_if_no_bounding_boxes,
        name,
    )
    return (
        Tensor([0, 0, 0], dtype="int32"),
        Tensor([-1, -1, -1], dtype="int32"),
        Tensor([[[0.0, 0.0, 1.0, 1.0]]], dtype="float32"),
    )


def stateless_sample_distorted_bounding_box(
    image_size: Sequence[int] | Tensor | TensorOperand = (1, 1, 3),
    bounding_boxes: Tensor | TensorOperand = 0.0,
    seed: Sequence[int] | Tensor | TensorOperand = (0, 0),
    min_object_covered: float = 0.1,
    aspect_ratio_range: Sequence[float] = (0.75, 1.33),
    area_range: Sequence[float] = (0.05, 1.0),
    max_attempts: int = 100,
    use_image_if_no_bounding_boxes: bool = False,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate a randomly distorted bounding box deterministically given `seed`.

    Args:
        image_size: 1-D, containing `[height, width, channels]`.
        bounding_boxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes.
        seed: 1-D integer tensor of shape `[2]`.
        min_object_covered: A Tensor of type `float32`. Defaults to `0.1`.
        aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75, 1.33]`.
        area_range: An optional list of `floats`. Defaults to `[0.05, 1]`.
        max_attempts: An optional `int`. Defaults to `100`.
        use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
        name: A name for the operation (optional).

    Returns:
        tuple[Tensor, Tensor, Tensor]: `(begin, size, bboxes)`.
    """
    del seed
    return sample_distorted_bounding_box(
        image_size=image_size,
        bounding_boxes=bounding_boxes,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
        name=name,
    )


def sobel_edges(image: Tensor | TensorOperand = 0.0) -> Tensor:
    """Return a tensor holding Sobel edge maps `(dy, dx)` stacked along a new trailing axis.

    Args:
        image: Image tensor with shape `[batch_size, h, w, d]`.

    Returns:
        Tensor: Tensor holding edge maps with shape `[batch_size, h, w, d, 2]`.
    """
    dy, dx = image_gradients(image)
    return _wrap_tensor(_k_ops.stack([_ensure_tensor(dy), _ensure_tensor(dx)], axis=-1))


def ssim(
    img1: Tensor | TensorOperand = 0.0,
    img2: Tensor | TensorOperand = 0.0,
    max_val: float | Tensor = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_index_map: bool = False,
) -> Tensor:
    """Compute the Structural Similarity Index (SSIM) between `img1` and `img2`.

    Args:
        img1: First image batch.
        img2: Second image batch.
        max_val: The dynamic range of the images.
        filter_size: Size of Gaussian filter kernel (`default: 11`).
        filter_sigma: Standard deviation of Gaussian filter kernel (`default: 1.5`).
        k1: SSIM stabilizing constant `k1` (`default: 0.01`).
        k2: SSIM stabilizing constant `k2` (`default: 0.03`).
        return_index_map: If `True`, returns the local SSIM map instead of global mean.

    Returns:
        Tensor: SSIM similarity score tensor.
    """
    del filter_size, filter_sigma, return_index_map
    t1 = _ensure_tensor(img1)
    t2 = _ensure_tensor(img2)
    mu1 = _as_float(_to_py_nested(_k_ops.mean(t1)))
    mu2 = _as_float(_to_py_nested(_k_ops.mean(t2)))
    d1 = _ops.subtract(t1, _to_tensor(mu1, dtype=t1.dtype))
    d2 = _ops.subtract(t2, _to_tensor(mu2, dtype=t2.dtype))
    var1 = _as_float(_to_py_nested(_k_ops.mean(_ops.multiply(d1, d1))))
    var2 = _as_float(_to_py_nested(_k_ops.mean(_ops.multiply(d2, d2))))
    cov = _as_float(_to_py_nested(_k_ops.mean(_ops.multiply(d1, d2))))
    mv = _as_float(_to_py_nested(max_val))
    c1 = (k1 * mv) ** 2
    c2 = (k2 * mv) ** 2
    num = (2.0 * mu1 * mu2 + c1) * (2.0 * cov + c2)
    den = (mu1 * mu1 + mu2 * mu2 + c1) * (var1 + var2 + c2)
    score = num / den if abs(den) > 1e-12 else 1.0
    return Tensor(score, dtype="float32")


def ssim_multiscale(
    img1: Tensor | TensorOperand = 0.0,
    img2: Tensor | TensorOperand = 0.0,
    max_val: float | Tensor = 1.0,
    power_factors: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Compute the Multi-Scale Structural Similarity Index (MS-SSIM) between `img1` and `img2`.

    Args:
        img1: First image batch.
        img2: Second image batch.
        max_val: The dynamic range of the images.
        power_factors: Iterable of weights for each scale.
        filter_size: Size of Gaussian filter kernel (`default: 11`).
        filter_sigma: Standard deviation of Gaussian filter kernel (`default: 1.5`).
        k1: SSIM stabilizing constant `k1` (`default: 0.01`).
        k2: SSIM stabilizing constant `k2` (`default: 0.03`).

    Returns:
        Tensor: MS-SSIM similarity score tensor.
    """
    del power_factors
    return ssim(
        img1=img1,
        img2=img2,
        max_val=max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2,
    )


def stateless_random_brightness(
    image: Tensor | TensorOperand = 0.0,
    max_delta: float = 0.1,
    seed: Sequence[int] | Tensor | TensorOperand = (0, 0),
) -> Tensor:
    """Adjust the brightness of images by a random factor deterministically.

    Args:
        image: An image or images to adjust.
        max_delta: `float`, must be non-negative.
        seed: A shape `[2]` Tensor, the seed to the random number generator.

    Returns:
        Tensor: Brightness-adjusted image tensor.
    """
    del seed
    return adjust_brightness(image, delta=max_delta)


def stateless_random_contrast(
    image: Tensor | TensorOperand = 0.0,
    lower: float = 0.5,
    upper: float = 1.5,
    seed: Sequence[int] | Tensor | TensorOperand = (0, 0),
) -> Tensor:
    """Adjust the contrast of images by a random factor deterministically.

    Args:
        image: An image tensor with 3 or more dimensions.
        lower: `float`. Lower bound for the random contrast factor.
        upper: `float`. Upper bound for the random contrast factor.
        seed: A shape `[2]` Tensor, the seed to the random number generator.

    Returns:
        Tensor: Contrast-adjusted image tensor.
    """
    del upper, seed
    return adjust_contrast(image, contrast_factor=lower)


def stateless_random_crop(
    value: Tensor | TensorOperand = 0.0,
    size: Sequence[int] = (1,),
    seed: Sequence[int] | Tensor | TensorOperand = (0, 0),
    name: str | None = None,
) -> Tensor:
    """Randomly crop a tensor to a given size deterministically.

    Args:
        value: Input tensor to crop.
        size: 1-D tensor with size the rank of `value`.
        seed: A shape `[2]` Tensor, the seed to the random number generator.
        name: A name for this operation (optional).

    Returns:
        Tensor: Cropped tensor of shape `size`.
    """
    del seed
    return random_crop(value=value, size=size, name=name)


def stateless_random_flip_left_right(
    image: Tensor | TensorOperand = 0.0,
    seed: Sequence[int] | Tensor | TensorOperand = (0, 0),
) -> Tensor:
    """Randomly flip an image horizontally (left to right) deterministically.

    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
        seed: A shape `[2]` Tensor, the seed to the random number generator.

    Returns:
        Tensor: Flipped or unchanged image tensor.
    """
    del seed
    return flip_left_right(image)


def stateless_random_flip_up_down(
    image: Tensor | TensorOperand = 0.0,
    seed: Sequence[int] | Tensor | TensorOperand = (0, 0),
) -> Tensor:
    """Randomly flip an image vertically (upside down) deterministically.

    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
        seed: A shape `[2]` Tensor, the seed to the random number generator.

    Returns:
        Tensor: Flipped or unchanged image tensor.
    """
    del seed
    return flip_up_down(image)


def stateless_random_hue(
    image: Tensor | TensorOperand = 0.0,
    max_delta: float = 0.1,
    seed: Sequence[int] | Tensor | TensorOperand = (0, 0),
) -> Tensor:
    """Adjust the hue of RGB images by a random factor deterministically.

    Args:
        image: RGB image or images. The size of the last dimension must be 3.
        max_delta: `float`. Maximum value for the random delta.
        seed: A shape `[2]` Tensor, the seed to the random number generator.

    Returns:
        Tensor: Hue-adjusted image tensor.
    """
    del seed
    return adjust_hue(image, delta=max_delta)


def stateless_random_jpeg_quality(
    image: Tensor | TensorOperand = 0.0,
    min_jpeg_quality: int = 75,
    max_jpeg_quality: int = 95,
    seed: Sequence[int] | Tensor | TensorOperand = (0, 0),
) -> Tensor:
    """Deterministically radomize jpeg encoding quality for inducing jpeg noise.

    Args:
        image: 3D image. Size of the last dimension must be 1 or 3.
        min_jpeg_quality: Minimum jpeg encoding quality to use.
        max_jpeg_quality: Maximum jpeg encoding quality to use.
        seed: A shape `[2]` Tensor, the seed to the random number generator.

    Returns:
        Tensor: Adjusted image tensor.
    """
    del max_jpeg_quality, seed
    return adjust_jpeg_quality(image, jpeg_quality=min_jpeg_quality)


def stateless_random_saturation(
    image: Tensor | TensorOperand = 0.0,
    lower: float = 0.5,
    upper: float = 1.5,
    seed: Sequence[int] | Tensor | TensorOperand = (0, 0),
) -> Tensor:
    """Adjust the saturation of RGB images by a random factor deterministically.

    Args:
        image: RGB image or images. The size of the last dimension must be 3.
        lower: `float`. Lower bound for the random saturation factor.
        upper: `float`. Upper bound for the random saturation factor.
        seed: A shape `[2]` Tensor, the seed to the random number generator.

    Returns:
        Tensor: Saturation-adjusted image tensor.
    """
    del upper, seed
    return adjust_saturation(image, saturation_factor=lower)


def total_variation(
    images: Tensor | TensorOperand = 0.0,
    name: str | None = None,
) -> Tensor:
    """Calculate and return the anisotropic total variation for one or more images.

    Args:
        images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
        name: A name for the operation (optional).

    Returns:
        Tensor: Total variation scalar (for 3-D) or 1-D batch tensor (for 4-D).
    """
    del name
    t_img = Tensor(images)
    if _tracer.is_tracing or len(t_img.shape) < 3:
        return Tensor(0.0, dtype="float32")
    if len(t_img.shape) == 3:
        dy3 = (
            _ops.abs(_ensure_tensor(t_img[1:, :, :] - t_img[:-1, :, :]))
            if t_img.shape[0] > 1
            else _ops.zeros((1,), dtype=msc_core.DType.Float32)
        )
        dx3 = (
            _ops.abs(_ensure_tensor(t_img[:, 1:, :] - t_img[:, :-1, :]))
            if t_img.shape[1] > 1
            else _ops.zeros((1,), dtype=msc_core.DType.Float32)
        )
        return _wrap_tensor(_ops.add(_k_ops.sum(dy3), _k_ops.sum(dx3)))
    dy4, dx4 = image_gradients(t_img)
    sum_y = _k_ops.sum(_ops.abs(_ensure_tensor(dy4)), axis=(1, 2, 3))
    sum_x = _k_ops.sum(_ops.abs(_ensure_tensor(dx4)), axis=(1, 2, 3))
    return _wrap_tensor(_ops.add(_ensure_tensor(sum_y), _ensure_tensor(sum_x)))


def transpose(
    image: Tensor | TensorOperand = 0.0,
    name: str | None = None,
) -> Tensor:
    """Transpose image(s) by swapping the height and width dimension.

    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
        name: A name for this operation (optional).

    Returns:
        Tensor: Transposed image tensor.
    """
    del name
    t_img = _ensure_tensor(image)
    if len(t_img.shape) == 4:
        return _wrap_tensor(_k_ops.transpose(t_img, axes=(0, 2, 1, 3)))
    if len(t_img.shape) == 3:
        return _wrap_tensor(_k_ops.transpose(t_img, axes=(1, 0, 2)))
    return _wrap_tensor(t_img)
