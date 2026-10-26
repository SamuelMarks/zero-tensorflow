"""TensorFlow `math` namespace and top-level mathematical/array operations."""

from __future__ import annotations

import builtins
import itertools
import math as _py_math
import struct
from collections.abc import Sequence

from zero_tensorflow._typing import (
    DTypeLike,
    NestedTensorData,
    ShapeLike,
    TensorCompatible,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.array_ops import (
    _flatten_floats,
    _flatten_ints,
    _to_py_data,
    tensor_scatter_nd_add,
)
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap, _wrap_tensor, to_dtype
from zero_tensorflow.dtypes import DType

import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from zero_keras import ops as _k_ops

__all__ = [
    "math",
    "abs",
    "acos",
    "acosh",
    "add",
    "argmax",
    "argmin",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitcast",
    "broadcast_to",
    "cast",
    "concat",
    "cos",
    "cosh",
    "divide",
    "einsum",
    "equal",
    "exp",
    "expand_dims",
    "eye",
    "fill",
    "floor",
    "gather",
    "gather_nd",
    "greater",
    "greater_equal",
    "identity",
    "less",
    "less_equal",
    "linspace",
    "logical_and",
    "logical_not",
    "logical_or",
    "matmul",
    "maximum",
    "meshgrid",
    "minimum",
    "multiply",
    "negative",
    "norm",
    "not_equal",
    "ones",
    "ones_like",
    "pow",
    "range",
    "reduce_all",
    "reduce_any",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_prod",
    "reduce_sum",
    "repeat",
    "reshape",
    "roll",
    "round",
    "scalar_mul",
    "scatter_nd",
    "shape",
    "sign",
    "sin",
    "sinh",
    "slice",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "strided_slice",
    "subtract",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "transpose",
    "truediv",
    "unstack",
    "where",
    "zeros",
    "zeros_like",
]


def _normalize_axis(
    axis: int | Sequence[int] | Tensor | TensorOperand,
) -> int | builtins.tuple[int, ...] | None:
    """Normalize a reduction or operation `axis` into an `int`, `tuple[int, ...]`, or `None`.

    Args:
        axis: Raw axis argument (integer, sequence of integers, 0-D/1-D Tensor, or None).

    Returns:
        int | tuple[int, ...] | None: Normalized axis value.
    """
    if axis is None:
        return None
    if isinstance(axis, int) and not isinstance(axis, builtins.bool):
        return axis
    if isinstance(axis, Tensor):
        raw = axis.tolist()
        if isinstance(raw, (int, float)):
            return int(raw)
        ints = [int(x) for x in _flatten_ints(raw)]
        return builtins.tuple(ints) if ints else None
    if isinstance(axis, (builtins.list, builtins.tuple)):
        ints_seq = [int(x) for x in axis if isinstance(x, int) and not isinstance(x, builtins.bool)]
        return builtins.tuple(ints_seq) if ints_seq else None
    return None


def _scalar_dawsn(x: float) -> float:
    """Compute Dawson's integral `D_+(x) = exp(-x^2) * int_0^x exp(t^2) dt` for a scalar float.

    Args:
        x: Real input scalar.

    Returns:
        float: Evaluated Dawson integral value.
    """
    if x == 0.0 or _py_math.isnan(x):
        return x
    sign = -1.0 if x < 0.0 else 1.0
    ax = builtins.abs(x)
    if ax <= 6.0:
        x2 = ax * ax
        term = ax
        total = ax
        for n in builtins.range(1, 85):
            term *= x2 / float(n)
            total += term / float(2 * n + 1)
        return sign * _py_math.exp(-x2) * total
    inv_2x2 = 0.5 / (ax * ax)
    asymp = 1.0
    term = 1.0
    for k in builtins.range(1, 8):
        term *= float(2 * k - 1) * inv_2x2
        asymp += term
    return sign * (0.5 / ax) * asymp


def _scalar_fresnel(x: float) -> builtins.tuple[float, float]:
    """Compute the Fresnel cosine and sine integrals `(C(x), S(x))` for a scalar float.

    Args:
        x: Real input scalar.

    Returns:
        tuple[float, float]: `(fresnel_cos(x), fresnel_sin(x))`.
    """
    if x == 0.0 or _py_math.isnan(x):
        return x, x
    sign = -1.0 if x < 0.0 else 1.0
    ax = builtins.abs(x)
    half_pi = 0.5 * _py_math.pi
    if ax <= 2.5:
        x2 = ax * ax
        x4 = x2 * x2
        c_term = ax
        c_sum = ax
        s_term = ax * x2 * half_pi
        s_sum = s_term / 3.0
        hp2 = half_pi * half_pi
        for n in builtins.range(1, 45):
            c_term *= -hp2 * x4 / float((2 * n - 1) * (2 * n))
            c_sum += c_term / float(4 * n + 1)
            s_term *= -hp2 * x4 / float((2 * n) * (2 * n + 1))
            s_sum += s_term / float(4 * n + 3)
        return sign * c_sum, sign * s_sum
    u = _py_math.pi * ax * ax
    inv_u2 = 1.0 / (u * u)
    f_sum = 1.0
    f_term = 1.0
    g_sum = 1.0 / u
    g_term = 1.0 / u
    for k in builtins.range(1, 5):
        f_term = -f_term * float((4 * k - 3) * (4 * k - 1)) * inv_u2
        f_sum += f_term
        g_term = -g_term * float((4 * k - 1) * (4 * k + 1)) * inv_u2
        g_sum += g_term
    f_val = f_sum / (_py_math.pi * ax)
    g_val = g_sum / (_py_math.pi * ax)
    theta = half_pi * ax * ax
    sin_t = _py_math.sin(theta)
    cos_t = _py_math.cos(theta)
    c_val = 0.5 + f_val * sin_t - g_val * cos_t
    s_val = 0.5 - f_val * cos_t - g_val * sin_t
    return sign * c_val, sign * s_val


def _scalar_spence(x: float) -> float:
    """Compute Spence's dilogarithm `Li_2(1 - x) = int_1^x ln(t)/(1 - t) dt` for `x >= 0`.

    Args:
        x: Real input scalar.

    Returns:
        float: Evaluated Spence function value.
    """
    if x < 0.0 or _py_math.isnan(x):
        return float("nan")
    if x == 0.0:
        return _py_math.pi * _py_math.pi / 6.0
    if x == 1.0:
        return 0.0
    if x > 2.0:
        ln_x = _py_math.log(x)
        return -_scalar_spence(1.0 / x) - 0.5 * ln_x * ln_x
    if x < 0.5:
        return (
            (_py_math.pi * _py_math.pi / 6.0)
            - _py_math.log(x) * _py_math.log(1.0 - x)
            - _scalar_spence(1.0 - x)
        )
    z = 1.0 - x
    s_z = 0.0
    zk = z
    for k in builtins.range(1, 60):
        s_z += zk / float(k * k)
        zk *= z
    return s_z


def _apply_scalar_fn(
    x: TensorCompatible | TensorOperand,
    fn_mode: str,
) -> Tensor:
    """Apply a scalar special function (`dawsn`, `fresnel_cos`, `fresnel_sin`, `spence`, `lgamma`) across an N-D tensor.

    Args:
        x: Input tensor or tensor-compatible value.
        fn_mode: Function identifier string.

    Returns:
        Tensor: Output tensor with identical shape and floating-point dtype.
    """
    t_x = _to_tensor(0.0 if x is None else x)
    out_dt = to_dtype(t_x.dtype)
    if not out_dt.is_floating:
        out_dt = DType.Float32
    vals = _flatten_floats(_to_py_data(t_x))
    mapped: builtins.list[float] = []
    for v in vals:
        if fn_mode == "dawsn":
            mapped.append(_scalar_dawsn(v))
        elif fn_mode == "fresnel_cos":
            mapped.append(_scalar_fresnel(v)[0])
        elif fn_mode == "fresnel_sin":
            mapped.append(_scalar_fresnel(v)[1])
        elif fn_mode == "spence":
            mapped.append(_scalar_spence(v))
        else:
            mapped.append(_py_math.lgamma(v))
    if not t_x.shape:
        return Tensor(mapped[0] if mapped else 0.0, dtype=out_dt)
    return _wrap_tensor(_k_ops.reshape(_to_tensor(mapped, dtype=out_dt), t_x.shape))


def _cumulative_nd(
    x: TensorCompatible | TensorOperand,
    axis: int | Tensor | TensorOperand,
    exclusive: builtins.bool,
    reverse: builtins.bool,
    mode: str,
) -> Tensor:
    """Compute cumulative sum, product, or logsumexp along `axis` with `exclusive` and `reverse` support.

    Args:
        x: Input tensor.
        axis: Axis along which to accumulate.
        exclusive: If True, exclude the current element from each prefix accumulation.
        reverse: If True, accumulate in reverse order along `axis`.
        mode: Accumulation mode (`'sum'`, `'prod'`, or `'logsumexp'`).

    Returns:
        Tensor: Cumulative result tensor of the same shape and dtype as `x`.
    """
    t_x = _to_tensor(0.0 if x is None else x)
    ax_norm = _normalize_axis(axis)
    eff_axis = ax_norm if isinstance(ax_norm, int) else (ax_norm[0] if ax_norm else 0)
    if _tracer.is_tracing:
        if mode == "sum":
            return _wrap_tensor(_ops.cumsum(t_x, axis=eff_axis))
        if mode == "prod":
            return _wrap_tensor(_ops.cumprod(t_x, axis=eff_axis))
        return _wrap_tensor(_ops.cumulative_logsumexp(t_x, axis=eff_axis))
    rank = len(t_x.shape)
    if eff_axis < 0:
        eff_axis += rank
    if not exclusive and not reverse and mode in ("sum", "prod"):
        if mode == "sum":
            return _wrap_tensor(_ops.cumsum(t_x, axis=eff_axis))
        return _wrap_tensor(_ops.cumprod(t_x, axis=eff_axis))
    dim_n = t_x.shape[eff_axis]
    slices = [_to_tensor(_k_ops.take(t_x, i, axis=eff_axis)) for i in builtins.range(dim_n)]
    order = (
        builtins.list(builtins.range(dim_n - 1, -1, -1))
        if reverse
        else builtins.list(builtins.range(dim_n))
    )
    out_slices: builtins.list[Tensor] = [_wrap_tensor(slices[0])] * dim_n
    if mode == "sum":
        acc = _to_tensor(_k_ops.zeros_like(slices[0]))
    elif mode == "prod":
        acc = _to_tensor(_k_ops.ones_like(slices[0]))
    else:
        acc = _to_tensor(_k_ops.full(slices[0].shape, float("-inf"), dtype=t_x.dtype))
    has_acc = False
    for idx in order:
        cur = slices[idx]
        if exclusive:
            out_slices[idx] = _wrap_tensor(acc)
        if mode == "sum":
            acc = _ops.add(acc, cur)
        elif mode == "prod":
            acc = _ops.multiply(acc, cur)
        else:
            if not has_acc:
                acc = cur
                has_acc = True
            else:
                m = _ops.maximum(acc, cur)
                acc = _ops.add(
                    m,
                    _ops.log(
                        _ops.add(
                            _ops.exp(_ops.subtract(acc, m)),
                            _ops.exp(_ops.subtract(cur, m)),
                        )
                    ),
                )
        if not exclusive:
            out_slices[idx] = _wrap_tensor(acc)
    return _wrap_tensor(_k_ops.stack([_to_tensor(s) for s in out_slices], axis=eff_axis))


def _segment_reduce_nd(
    data: TensorCompatible | TensorOperand,
    segment_ids: TensorCompatible | TensorOperand,
    num_segments: int | TensorCompatible | TensorOperand,
    mode: str,
    is_unsorted: builtins.bool,
) -> Tensor:
    """Compute sorted or unsorted segment reduction preserving trailing N-D dimensions.

    Args:
        data: N-D input data tensor.
        segment_ids: Segment index tensor of rank `K >= 1`.
        num_segments: Explicit segment count (or `None` to infer from `max(segment_ids) + 1`).
        mode: Reduction mode (`'sum'`, `'mean'`, `'prod'`, `'min'`, `'max'`, `'sqrt_n'`).
        is_unsorted: Whether called from an `unsorted_segment_*` op.

    Returns:
        Tensor: Reduced tensor of shape `(num_segments, *data.shape[segment_ids.ndim:])`.
    """
    t_data = _to_tensor(0.0 if data is None else data)
    t_seg = _to_tensor([0] if segment_ids is None else segment_ids, dtype=DType.Int32)
    seg_list = _flatten_ints(_to_py_data(t_seg))
    if isinstance(num_segments, Tensor):
        raw_ns = num_segments.tolist()
        n_seg = int(raw_ns) if isinstance(raw_ns, (int, float)) else 1
    elif isinstance(num_segments, (int, float)):
        n_seg = int(num_segments)
    else:
        n_seg = (builtins.max(seg_list) + 1) if seg_list else 0
    n_seg = builtins.max(1, n_seg)
    if _tracer.is_tracing:
        if is_unsorted:
            if mode == "sum":
                return _wrap_tensor(as_operand(_ops.unsorted_segment_sum(t_data, t_seg, n_seg)))
            if mode == "mean":
                return _wrap_tensor(as_operand(_ops.unsorted_segment_mean(t_data, t_seg, n_seg)))
            if mode == "prod":
                return _wrap_tensor(as_operand(_ops.unsorted_segment_prod(t_data, t_seg, n_seg)))
            if mode == "min":
                return _wrap_tensor(as_operand(_ops.unsorted_segment_min(t_data, t_seg, n_seg)))
            if mode == "max":
                return _wrap_tensor(as_operand(_ops.unsorted_segment_max(t_data, t_seg, n_seg)))
            return _wrap_tensor(as_operand(_ops.unsorted_segment_sqrt_n(t_data, t_seg, n_seg)))
        if mode == "sum":
            return _wrap_tensor(as_operand(_ops.segment_sum(t_data, t_seg)))
        if mode == "mean":
            return _wrap_tensor(as_operand(_ops.segment_mean(t_data, t_seg)))
        if mode == "prod":
            return _wrap_tensor(as_operand(_ops.segment_prod(t_data, t_seg)))
        if mode == "min":
            return _wrap_tensor(as_operand(_ops.segment_min(t_data, t_seg)))
        return _wrap_tensor(as_operand(_ops.segment_max(t_data, t_seg)))
    seg_rank = builtins.max(1, len(t_seg.shape))
    inner_shape = t_data.shape[seg_rank:] if len(t_data.shape) >= seg_rank else ()
    n_items = len(seg_list)
    flat_data = _to_tensor(_k_ops.reshape(t_data, (builtins.max(1, n_items), *inner_shape)))
    buckets: builtins.list[builtins.list[int]] = [[] for _ in builtins.range(n_seg)]
    for i, s_id in enumerate(seg_list):
        if 0 <= s_id < n_seg:
            buckets[s_id].append(i)
    zero_slice = _to_tensor(_k_ops.zeros(inner_shape, dtype=t_data.dtype))
    one_slice = _to_tensor(_k_ops.ones(inner_shape, dtype=t_data.dtype))
    out_rows: builtins.list[Tensor] = []
    for s_id in builtins.range(n_seg):
        idxs = buckets[s_id]
        if not idxs:
            out_rows.append(_wrap_tensor(one_slice if mode == "prod" else zero_slice))
            continue
        first_row = _to_tensor(_k_ops.take(flat_data, idxs[0], axis=0))
        acc = first_row
        for j in idxs[1:]:
            row_j = _to_tensor(_k_ops.take(flat_data, j, axis=0))
            if mode in ("sum", "mean", "sqrt_n"):
                acc = _ops.add(acc, row_j)
            elif mode == "prod":
                acc = _ops.multiply(acc, row_j)
            elif mode == "min":
                acc = _ops.minimum(acc, row_j)
            else:
                acc = _ops.maximum(acc, row_j)
        if mode == "mean":
            acc = _ops.divide(acc, _to_tensor(float(len(idxs)), dtype=t_data.dtype))
        elif mode == "sqrt_n":
            acc = _ops.divide(acc, _to_tensor(_py_math.sqrt(float(len(idxs))), dtype=t_data.dtype))
        out_rows.append(_wrap_tensor(acc))
    return _wrap_tensor(_k_ops.stack([_to_tensor(r) for r in out_rows], axis=0))


class math:
    """TensorFlow `math` namespace."""

    @staticmethod
    def abs(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.abs` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.abs(_to_tensor(x)))

    @staticmethod
    def acos(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.acos` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.acos(_to_tensor(x)))

    @staticmethod
    def acosh(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.acosh` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.acosh(_to_tensor(x)))

    @staticmethod
    def asin(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.asin` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.asin(_to_tensor(x)))

    @staticmethod
    def asinh(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.asinh` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.asinh(_to_tensor(x)))

    @staticmethod
    def atan(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.atan` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.atan(_to_tensor(x)))

    @staticmethod
    def atanh(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.atanh` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.atanh(_to_tensor(x)))

    @staticmethod
    def bessel_i0(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.bessel_i0` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.bessel_i0(_to_tensor(x)))

    @staticmethod
    def bessel_i0e(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.bessel_i0e` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.bessel_i0e(_to_tensor(x)))

    @staticmethod
    def bessel_i1(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.bessel_i1` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.bessel_i1(_to_tensor(x)))

    @staticmethod
    def bessel_i1e(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.bessel_i1e` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.bessel_i1e(_to_tensor(x)))

    @staticmethod
    def ceil(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.ceil` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.ceil(_to_tensor(x)))

    @staticmethod
    def conj(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.conj` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.conj(_to_tensor(x)))

    @staticmethod
    def cos(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.cos` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.cos(_to_tensor(x)))

    @staticmethod
    def cosh(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.cosh` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.cosh(_to_tensor(x)))

    @staticmethod
    def digamma(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.digamma` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.digamma(_to_tensor(x)))

    @staticmethod
    def erf(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.erf` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.erf(_to_tensor(x)))

    @staticmethod
    def erfc(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.erfc` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.erfc(_to_tensor(x)))

    @staticmethod
    def erfinv(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.erfinv` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.erfinv(_to_tensor(x)))

    @staticmethod
    def exp(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.exp` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.exp(_to_tensor(x)))

    @staticmethod
    def expm1(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.expm1` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.expm1(_to_tensor(x)))

    @staticmethod
    def floor(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.floor` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.floor(_to_tensor(x)))

    @staticmethod
    def is_finite(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.is_finite` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.isfinite(_to_tensor(x)))

    @staticmethod
    def is_inf(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.is_inf` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.isinf(_to_tensor(x)))

    @staticmethod
    def is_nan(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.is_nan` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.isnan(_to_tensor(x)))

    @staticmethod
    def is_non_decreasing(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.is_non_decreasing` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.is_non_decreasing(_to_tensor(x)))

    @staticmethod
    def is_strictly_increasing(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.is_strictly_increasing` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.is_strictly_increasing(_to_tensor(x)))

    @staticmethod
    def log(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.log` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.log(_to_tensor(x)))

    @staticmethod
    def log1p(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.log1p` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.log1p(_to_tensor(x)))

    @staticmethod
    def log_sigmoid(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.log_sigmoid` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_k_ops.log_sigmoid(_to_tensor(x)))

    @staticmethod
    def logical_not(
        x: TensorCompatible | TensorOperand = True,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.logical_not` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.logical_not(_to_tensor(x)))

    @staticmethod
    def negative(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.negative` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.negative(_to_tensor(x)))

    @staticmethod
    def reciprocal(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reciprocal` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.reciprocal(_to_tensor(x)))

    @staticmethod
    def reciprocal_no_nan(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reciprocal_no_nan` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.reciprocal_no_nan(_to_tensor(x)))

    @staticmethod
    def rint(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.rint` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.rint(_to_tensor(x)))

    @staticmethod
    def round(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.round` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.round(_to_tensor(x)))

    @staticmethod
    def rsqrt(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.rsqrt` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.rsqrt(_to_tensor(x)))

    @staticmethod
    def sigmoid(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.sigmoid` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_k_ops.sigmoid(_to_tensor(x)))

    @staticmethod
    def sign(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.sign` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.sign(_to_tensor(x)))

    @staticmethod
    def sin(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.sin` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.sin(_to_tensor(x)))

    @staticmethod
    def sinh(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.sinh` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.sinh(_to_tensor(x)))

    @staticmethod
    def sqrt(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.sqrt` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.sqrt(_to_tensor(x)))

    @staticmethod
    def square(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.square` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.square(_to_tensor(x)))

    @staticmethod
    def tan(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.tan` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.tan(_to_tensor(x)))

    @staticmethod
    def tanh(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.tanh` operation.

        Args:
            x: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.tanh(_to_tensor(x)))

    @staticmethod
    def angle(
        input: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.angle` operation.

        Args:
            input: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.angle(_to_tensor(input)))

    @staticmethod
    def imag(
        input: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.imag` operation.

        Args:
            input: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.imag(_to_tensor(input)))

    @staticmethod
    def real(
        input: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.real` operation.

        Args:
            input: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.real(_to_tensor(input)))

    @staticmethod
    def softplus(
        features: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.softplus` operation.

        Args:
            features: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_k_ops.softplus(_to_tensor(features)))

    @staticmethod
    def softsign(
        features: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.softsign` operation.

        Args:
            features: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_k_ops.softsign(_to_tensor(features)))

    @staticmethod
    def zero_fraction(
        value: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.zero_fraction` operation.

        Args:
            value: Input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(as_operand(_ops.zero_fraction(_to_tensor(value))))

    @staticmethod
    def add(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.add` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.add(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def divide(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.divide` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.divide(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def divide_no_nan(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.divide_no_nan` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(as_operand(_ops.divide_no_nan(_to_tensor(x), _to_tensor(y))))

    @staticmethod
    def equal(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.equal` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.equal(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def floordiv(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.floordiv` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.floor_divide(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def floormod(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.floormod` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.remainder(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def greater(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.greater` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.greater(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def greater_equal(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.greater_equal` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.greater_equal(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def less(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.less` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.less(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def less_equal(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.less_equal` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.less_equal(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def logical_and(
        x: TensorCompatible | TensorOperand = True,
        y: TensorCompatible | TensorOperand = True,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.logical_and` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.logical_and(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def logical_or(
        x: TensorCompatible | TensorOperand = True,
        y: TensorCompatible | TensorOperand = True,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.logical_or` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.logical_or(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def logical_xor(
        x: TensorCompatible | TensorOperand = True,
        y: TensorCompatible | TensorOperand = True,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.logical_xor` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.logical_xor(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def maximum(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.maximum` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.maximum(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def minimum(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.minimum` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.minimum(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def mod(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.mod` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.remainder(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def multiply(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.multiply` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.multiply(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def multiply_no_nan(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.multiply_no_nan` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.multiply_no_nan(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def not_equal(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.not_equal` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.not_equal(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def pow(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.pow` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.power(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def squared_difference(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.squared_difference` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.squared_difference(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def subtract(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.subtract` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.subtract(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def truediv(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.truediv` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.true_divide(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def xdivy(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.xdivy` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.xdivy(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def xlog1py(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.xlog1py` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.xlog1py(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def xlogy(
        x: TensorCompatible | TensorOperand = 1.0,
        y: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.xlogy` operation.

        Args:
            x: First input tensor operand.
            y: Second input tensor operand.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.xlogy(_to_tensor(x), _to_tensor(y)))

    @staticmethod
    def atan2(
        y: TensorCompatible | TensorOperand = 0.0,
        x: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.atan2` operation.

        Args:
            y: Operation parameter `y`.
            x: Operation parameter `x`.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.atan2(_to_tensor(y), _to_tensor(x)))

    @staticmethod
    def nextafter(
        x1: TensorCompatible | TensorOperand = 0.0,
        x2: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.nextafter` operation.

        Args:
            x1: Operation parameter `x1`.
            x2: Operation parameter `x2`.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.nextafter(_to_tensor(x1), _to_tensor(x2)))

    @staticmethod
    def igamma(
        a: TensorCompatible | TensorOperand = 1.0,
        x: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.igamma` operation.

        Args:
            a: Operation parameter `a`.
            x: Operation parameter `x`.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.igamma(_to_tensor(a), _to_tensor(x)))

    @staticmethod
    def igammac(
        a: TensorCompatible | TensorOperand = 1.0,
        x: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.igammac` operation.

        Args:
            a: Operation parameter `a`.
            x: Operation parameter `x`.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.igammac(_to_tensor(a), _to_tensor(x)))

    @staticmethod
    def polygamma(
        a: TensorCompatible | TensorOperand = 0.0,
        x: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.polygamma` operation.

        Args:
            a: Operation parameter `a`.
            x: Operation parameter `x`.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.polygamma(_to_tensor(a), _to_tensor(x)))

    @staticmethod
    def zeta(
        x: TensorCompatible | TensorOperand = 2.0,
        q: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.zeta` operation.

        Args:
            x: Operation parameter `x`.
            q: Operation parameter `q`.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(as_operand(_ops.zeta(_to_tensor(x), _to_tensor(q))))

    @staticmethod
    def scalar_mul(
        scalar: TensorCompatible | TensorOperand = 1.0,
        x: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.scalar_mul` operation.

        Args:
            scalar: Operation parameter `scalar`.
            x: Operation parameter `x`.
            name: Optional operation name.

        Returns:
            Tensor: Operation result.
        """
        del name
        return _wrap_tensor(_ops.multiply(_to_tensor(scalar), _to_tensor(x)))

    @staticmethod
    def reduce_all(
        input_tensor: TensorCompatible | TensorOperand = True,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_all` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.all(_to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims))
        )

    @staticmethod
    def reduce_any(
        input_tensor: TensorCompatible | TensorOperand = True,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_any` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.any(_to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims))
        )

    @staticmethod
    def reduce_euclidean_norm(
        input_tensor: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_euclidean_norm` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.reduce_euclidean_norm(
                _to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims)
            )
        )

    @staticmethod
    def reduce_logsumexp(
        input_tensor: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_logsumexp` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.reduce_logsumexp(
                _to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims)
            )
        )

    @staticmethod
    def reduce_max(
        input_tensor: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_max` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.max(_to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims))
        )

    @staticmethod
    def reduce_mean(
        input_tensor: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_mean` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.mean(_to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims))
        )

    @staticmethod
    def reduce_min(
        input_tensor: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_min` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.min(_to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims))
        )

    @staticmethod
    def reduce_prod(
        input_tensor: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_prod` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.prod(_to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims))
        )

    @staticmethod
    def reduce_std(
        input_tensor: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_std` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.std(_to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims))
        )

    @staticmethod
    def reduce_sum(
        input_tensor: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_sum` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.sum(_to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims))
        )

    @staticmethod
    def reduce_variance(
        input_tensor: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.reduce_variance` reduction operation.

        Args:
            input_tensor: Input tensor to reduce.
            axis: Dimensions along which to reduce.
            keepdims: Whether to retain reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Reduced tensor.
        """
        del name
        ax = _normalize_axis(axis)
        return _wrap_tensor(
            _ops.variance(_to_tensor(input_tensor), axis=ax, keepdims=builtins.bool(keepdims))
        )

    @staticmethod
    def argmax(
        input: TensorCompatible | TensorOperand = 0.0,
        axis: int | Tensor | TensorOperand = None,
        output_type: DTypeLike | TensorOperand = DType.Int64,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.argmax` operation.

        Args:
            input: Input tensor.
            axis: Axis along which to find the maximum index.
            output_type: Dtype for output indices (`int32` or `int64`).
            name: Optional operation name.

        Returns:
            Tensor: Tensor of maximum indices.
        """
        del output_type, name
        ax = _normalize_axis(axis)
        eff_ax = ax if isinstance(ax, int) else (ax[0] if ax else None)
        return _wrap_tensor(_ops.argmax(_to_tensor(input), axis=eff_ax))

    @staticmethod
    def argmin(
        input: TensorCompatible | TensorOperand = 0.0,
        axis: int | Tensor | TensorOperand = None,
        output_type: DTypeLike | TensorOperand = DType.Int64,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.argmin` operation.

        Args:
            input: Input tensor.
            axis: Axis along which to find the minimum index.
            output_type: Dtype for output indices (`int32` or `int64`).
            name: Optional operation name.

        Returns:
            Tensor: Tensor of minimum indices.
        """
        del output_type, name
        ax = _normalize_axis(axis)
        eff_ax = ax if isinstance(ax, int) else (ax[0] if ax else None)
        return _wrap_tensor(_ops.argmin(_to_tensor(input), axis=eff_ax))

    @staticmethod
    def count_nonzero(
        input: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        keepdims: builtins.bool | None = None,
        dtype: DTypeLike | TensorOperand = DType.Int64,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.count_nonzero` operation.

        Args:
            input: Input tensor.
            axis: Dimensions along which to count non-zero elements.
            keepdims: Whether to keep reduced dimensions.
            dtype: Output integer dtype.
            name: Optional operation name.

        Returns:
            Tensor: Count of non-zero elements.
        """
        del dtype, name
        ax = _normalize_axis(axis)
        kd = builtins.bool(keepdims) if keepdims is not None else False
        return _wrap_tensor(_ops.count_nonzero(_to_tensor(input), axis=ax, keepdims=kd))

    @staticmethod
    def accumulate_n(
        inputs: Sequence[TensorCompatible] | TensorCompatible | TensorOperand = (),
        shape: ShapeLike | TensorOperand = None,
        tensor_dtype: DTypeLike | TensorOperand = None,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.accumulate_n` elementwise sum of a list of tensors.

        Args:
            inputs: Sequence of input tensors.
            shape: Expected shape of input tensors.
            tensor_dtype: Expected dtype of input tensors.
            name: Optional operation name.

        Returns:
            Tensor: Elementwise sum of `inputs`.
        """
        del shape, tensor_dtype, name
        items = (
            [_to_tensor(t) for t in inputs]
            if isinstance(inputs, (builtins.list, builtins.tuple)) and len(inputs) > 0
            else [
                _to_tensor(
                    0.0
                    if inputs is None or isinstance(inputs, (builtins.list, builtins.tuple))
                    else inputs
                )
            ]
        )
        return _wrap_tensor(_ops.accumulate_n(items))

    @staticmethod
    def add_n(
        inputs: Sequence[TensorCompatible] | TensorCompatible | TensorOperand = (),
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.add_n` elementwise sum of a list of tensors.

        Args:
            inputs: Sequence of input tensors.
            name: Optional operation name.

        Returns:
            Tensor: Elementwise sum of `inputs`.
        """
        del name
        items = (
            [_to_tensor(t) for t in inputs]
            if isinstance(inputs, (builtins.list, builtins.tuple)) and len(inputs) > 0
            else [
                _to_tensor(
                    0.0
                    if inputs is None or isinstance(inputs, (builtins.list, builtins.tuple))
                    else inputs
                )
            ]
        )
        return _wrap_tensor(_ops.add_n(items))

    @staticmethod
    def approx_max_k(
        operand: TensorCompatible | TensorOperand = 0.0,
        k: int = 1,
        reduction_dimension: int = -1,
        recall_target: float = 0.95,
        reduction_input_size_override: int = -1,
        aggregate_to_topk: builtins.bool = True,
        name: str | None = None,
    ) -> builtins.tuple[Tensor, Tensor]:
        """Execute `math.approx_max_k` operation.

        Args:
            operand: Input tensor to search.
            k: Number of top elements to find.
            reduction_dimension: Dimension along which to search.
            recall_target: Recall target for the approximation.
            reduction_input_size_override: Input size override when evaluating subgraphs.
            aggregate_to_topk: Whether to aggregate results into sorted top-k.
            name: Optional operation name.

        Returns:
            tuple[Tensor, Tensor]: `(values, indices)` tuple.
        """
        del (
            reduction_dimension,
            recall_target,
            reduction_input_size_override,
            aggregate_to_topk,
            name,
        )
        v, i = _ops.approx_max_k(_to_tensor(operand), int(k))
        return _wrap_tensor(v), _wrap_tensor(i)

    @staticmethod
    def approx_min_k(
        operand: TensorCompatible | TensorOperand = 0.0,
        k: int = 1,
        reduction_dimension: int = -1,
        recall_target: float = 0.95,
        reduction_input_size_override: int = -1,
        aggregate_to_topk: builtins.bool = True,
        name: str | None = None,
    ) -> builtins.tuple[Tensor, Tensor]:
        """Execute `math.approx_min_k` operation.

        Args:
            operand: Input tensor to search.
            k: Number of minimum elements to find.
            reduction_dimension: Dimension along which to search.
            recall_target: Recall target for the approximation.
            reduction_input_size_override: Input size override when evaluating subgraphs.
            aggregate_to_topk: Whether to aggregate results into sorted bottom-k.
            name: Optional operation name.

        Returns:
            tuple[Tensor, Tensor]: `(values, indices)` tuple.
        """
        del (
            reduction_dimension,
            recall_target,
            reduction_input_size_override,
            aggregate_to_topk,
            name,
        )
        v, i = _ops.approx_min_k(_to_tensor(operand), int(k))
        return _wrap_tensor(v), _wrap_tensor(i)

    @staticmethod
    def betainc(
        a: TensorCompatible | TensorOperand = 1.0,
        b: TensorCompatible | TensorOperand = 1.0,
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Execute `math.betainc` regularized incomplete beta integral operation.

        Args:
            a: First positive shape parameter tensor.
            b: Second positive shape parameter tensor.
            x: Upper integration limit tensor in `[0, 1]`.
            name: Optional operation name.

        Returns:
            Tensor: Regularized incomplete beta values.
        """
        del name
        return _wrap_tensor(_ops.betainc(_to_tensor(a), _to_tensor(b), _to_tensor(x)))

    @staticmethod
    def bincount(
        arr: TensorCompatible | TensorOperand = (),
        weights: TensorCompatible | TensorOperand = None,
        minlength: int | TensorCompatible | TensorOperand = None,
        maxlength: int | TensorCompatible | TensorOperand = None,
        dtype: DTypeLike | TensorOperand = DType.Int32,
        name: str | None = None,
        axis: int | None = None,
        binary_output: builtins.bool = False,
    ) -> Tensor:
        """Count the number of occurrences of each non-negative integer value in `arr`.

        Args:
            arr: 1-D or 2-D integer tensor.
            weights: Optional weight tensor of the same shape as `arr`.
            minlength: Minimum number of bins in the output tensor.
            maxlength: Maximum number of bins in the output tensor.
            dtype: Output dtype when `weights` is None.
            name: Optional operation name.
            axis: Axis to bin along (`None`, `0`, or `-1` for 2-D row-wise bincount).
            binary_output: If True, output `1` for any bin with at least one occurrence.

        Returns:
            Tensor: 1-D or 2-D tensor of bin counts or weighted sums.
        """
        del name
        t_arr = _to_tensor(0 if arr is None else arr, dtype=DType.Int32)
        min_l = int(minlength) if isinstance(minlength, (int, float)) else 0
        max_l = int(maxlength) if isinstance(maxlength, (int, float)) else None
        if _tracer.is_tracing:
            w_t = _to_tensor(weights) if weights is not None else None
            return _wrap_tensor(_k_ops.bincount(t_arr, weights=w_t, minlength=min_l))
        t_w = _to_tensor(weights) if weights is not None and not binary_output else None
        out_dt = (
            to_dtype(t_w.dtype)
            if t_w is not None
            else to_dtype(dtype if dtype is not None else DType.Int32)
        )
        all_vals = [
            v for v in _flatten_ints(_to_py_data(t_arr)) if v >= 0 and (max_l is None or v < max_l)
        ]
        max_val = builtins.max(all_vals) if all_vals else -1
        n_bins = builtins.max(min_l, max_val + 1)
        if max_l is not None:
            n_bins = builtins.min(max_l, builtins.max(min_l, max_val + 1))
        if len(t_arr.shape) == 2 and axis == -1:
            n_rows = t_arr.shape[0]
            n_cols = t_arr.shape[1]
            w_flat = _flatten_floats(_to_py_data(t_w)) if t_w is not None else None
            arr_flat = _flatten_ints(_to_py_data(t_arr))
            rows_out: builtins.list[builtins.list[int | float]] = []
            for r in builtins.range(n_rows):
                bins_r: builtins.list[float] = [0.0] * n_bins
                for c in builtins.range(n_cols):
                    idx_flat = r * n_cols + c
                    val = arr_flat[idx_flat]
                    if 0 <= val < n_bins:
                        if binary_output:
                            bins_r[val] = 1.0
                        elif w_flat is not None:
                            bins_r[val] += w_flat[idx_flat]
                        else:
                            bins_r[val] += 1.0
                rows_out.append([int(b) if out_dt.is_integer else b for b in bins_r])
            return Tensor(rows_out, dtype=out_dt)
        arr_1d = _flatten_ints(_to_py_data(t_arr))
        w_1d = _flatten_floats(_to_py_data(t_w)) if t_w is not None else None
        bins_1d: builtins.list[float] = [0.0] * n_bins
        for i, val in enumerate(arr_1d):
            if 0 <= val < n_bins:
                if binary_output:
                    bins_1d[val] = 1.0
                elif w_1d is not None:
                    bins_1d[val] += w_1d[i]
                else:
                    bins_1d[val] += 1.0
        return Tensor([int(b) if out_dt.is_integer else b for b in bins_1d], dtype=out_dt)

    @staticmethod
    def confusion_matrix(
        labels: TensorCompatible | TensorOperand = (),
        predictions: TensorCompatible | TensorOperand = (),
        num_classes: int | TensorCompatible | TensorOperand = None,
        weights: TensorCompatible | TensorOperand = None,
        dtype: DTypeLike | TensorOperand = DType.Int32,
        name: str | None = None,
    ) -> Tensor:
        """Compute the confusion matrix from 1-D `labels` and `predictions` tensors.

        Args:
            labels: 1-D tensor of ground-truth class labels.
            predictions: 1-D tensor of predicted class labels.
            num_classes: Total number of classes (inferred from max label/prediction if None).
            weights: Optional 1-D tensor of per-sample weights.
            dtype: Data type of the confusion matrix (`int32` by default).
            name: Optional operation name.

        Returns:
            Tensor: 2-D `[num_classes, num_classes]` confusion matrix tensor.
        """
        del name
        t_lbl = _to_tensor([0] if labels is None else labels, dtype=DType.Int32)
        t_pred = _to_tensor([0] if predictions is None else predictions, dtype=DType.Int32)
        lbl_list = _flatten_ints(_to_py_data(t_lbl))
        pred_list = _flatten_ints(_to_py_data(t_pred))
        if isinstance(num_classes, Tensor):
            raw_nc = num_classes.tolist()
            n_cls = int(raw_nc) if isinstance(raw_nc, (int, float)) else 2
        elif isinstance(num_classes, (int, float)):
            n_cls = int(num_classes)
        else:
            combined = lbl_list + pred_list
            n_cls = (builtins.max(combined) + 1) if combined else 0
        if _tracer.is_tracing:
            return _wrap_tensor(
                as_operand(_ops.confusion_matrix(t_lbl, t_pred, builtins.max(1, n_cls)))
            )
        out_dt = to_dtype(dtype if dtype is not None else DType.Int32)
        mat: builtins.list[builtins.list[float]] = [
            [0.0 for _ in builtins.range(n_cls)] for _ in builtins.range(n_cls)
        ]
        if weights is not None:
            w_list = _flatten_floats(_to_py_data(_to_tensor(weights)))
            for r, c, w in zip(lbl_list, pred_list, w_list):
                if 0 <= r < n_cls and 0 <= c < n_cls:
                    mat[r][c] += w
        else:
            for r, c in zip(lbl_list, pred_list):
                if 0 <= r < n_cls and 0 <= c < n_cls:
                    mat[r][c] += 1.0
        typed_mat = [[int(val) if out_dt.is_integer else val for val in row] for row in mat]
        return Tensor(typed_mat, dtype=out_dt)

    @staticmethod
    def cumprod(
        x: TensorCompatible | TensorOperand = 1.0,
        axis: int | Tensor | TensorOperand = 0,
        exclusive: builtins.bool = False,
        reverse: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Compute the cumulative product of tensor `x` along `axis`.

        Args:
            x: Input tensor.
            axis: Axis along which to compute the cumulative product.
            exclusive: If True, perform exclusive cumulative product.
            reverse: If True, accumulate in reverse direction.
            name: Optional operation name.

        Returns:
            Tensor: Cumulative product tensor.
        """
        del name
        return _cumulative_nd(x, axis, builtins.bool(exclusive), builtins.bool(reverse), "prod")

    @staticmethod
    def cumsum(
        x: TensorCompatible | TensorOperand = 0.0,
        axis: int | Tensor | TensorOperand = 0,
        exclusive: builtins.bool = False,
        reverse: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Compute the cumulative sum of tensor `x` along `axis`.

        Args:
            x: Input tensor.
            axis: Axis along which to compute the cumulative sum.
            exclusive: If True, perform exclusive cumulative sum.
            reverse: If True, accumulate in reverse direction.
            name: Optional operation name.

        Returns:
            Tensor: Cumulative sum tensor.
        """
        del name
        return _cumulative_nd(x, axis, builtins.bool(exclusive), builtins.bool(reverse), "sum")

    @staticmethod
    def cumulative_logsumexp(
        x: TensorCompatible | TensorOperand = 0.0,
        axis: int | Tensor | TensorOperand = 0,
        exclusive: builtins.bool = False,
        reverse: builtins.bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Compute the cumulative log-sum-exp of tensor `x` along `axis`.

        Args:
            x: Input floating-point tensor.
            axis: Axis along which to accumulate.
            exclusive: If True, perform exclusive cumulative log-sum-exp.
            reverse: If True, accumulate in reverse direction.
            name: Optional operation name.

        Returns:
            Tensor: Cumulative log-sum-exp tensor.
        """
        del name
        return _cumulative_nd(
            x, axis, builtins.bool(exclusive), builtins.bool(reverse), "logsumexp"
        )

    @staticmethod
    def dawsn(
        x: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Compute Dawson's integral `exp(-x^2) * int_0^x exp(t^2) dt` elementwise.

        Args:
            x: Input floating-point tensor.
            name: Optional operation name.

        Returns:
            Tensor: Dawson integral tensor.
        """
        del name
        return _apply_scalar_fn(x, "dawsn")

    @staticmethod
    def erfcinv(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Compute the inverse complementary error function `erfinv(1 - x)` elementwise.

        Args:
            x: Input tensor in `(0, 2)`.
            name: Optional operation name.

        Returns:
            Tensor: Inverse complementary error function values.
        """
        del name
        t_x = _to_tensor(0.5 if x is None else x)
        if _tracer.is_tracing:
            return _wrap_tensor(_ops.erfcinv(t_x))
        return _wrap_tensor(_ops.erfinv(_ops.subtract(_to_tensor(1.0, dtype=t_x.dtype), t_x)))

    @staticmethod
    def fresnel_cos(
        x: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the Fresnel cosine integral `C(x) = int_0^x cos(pi/2 * t^2) dt` elementwise.

        Args:
            x: Input floating-point tensor.
            name: Optional operation name.

        Returns:
            Tensor: Fresnel cosine integral tensor.
        """
        del name
        return _apply_scalar_fn(x, "fresnel_cos")

    @staticmethod
    def Fresnel_cos(  # noqa: N802
        x: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the Fresnel cosine integral `C(x)` (alias for `math.fresnel_cos`).

        Args:
            x: Input floating-point tensor.
            name: Optional operation name.

        Returns:
            Tensor: Fresnel cosine integral tensor.
        """
        return math.fresnel_cos(x, name=name)

    @staticmethod
    def fresnel_sin(
        x: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the Fresnel sine integral `S(x) = int_0^x sin(pi/2 * t^2) dt` elementwise.

        Args:
            x: Input floating-point tensor.
            name: Optional operation name.

        Returns:
            Tensor: Fresnel sine integral tensor.
        """
        del name
        return _apply_scalar_fn(x, "fresnel_sin")

    @staticmethod
    def spence(
        x: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Compute Spence's dilogarithm `Li_2(1 - x) = int_1^x ln(t)/(1 - t) dt` elementwise.

        Args:
            x: Input non-negative floating-point tensor.
            name: Optional operation name.

        Returns:
            Tensor: Spence dilogarithm tensor.
        """
        del name
        return _apply_scalar_fn(x, "spence")

    @staticmethod
    def in_top_k(
        targets: TensorCompatible | TensorOperand = 0,
        predictions: TensorCompatible | TensorOperand = 0.0,
        k: int = 1,
        name: str | None = None,
    ) -> Tensor:
        """Check whether `targets` are in the top `k` `predictions`.

        Args:
            targets: 1-D tensor of class ids (`batch_size`).
            predictions: 2-D tensor of class scores (`[batch_size, num_classes]`).
            k: Number of top elements to look at.
            name: Optional operation name.

        Returns:
            Tensor: 1-D boolean tensor of shape `[batch_size]`.
        """
        del name
        return _wrap_tensor(_k_ops.in_top_k(_to_tensor(targets), _to_tensor(predictions), int(k)))

    @staticmethod
    def invert_permutation(
        x: TensorCompatible | TensorOperand = (),
        name: str | None = None,
    ) -> Tensor:
        """Compute the inverse permutation of a 1-D integer tensor `x`.

        Args:
            x: 1-D integer permutation tensor.
            name: Optional operation name.

        Returns:
            Tensor: 1-D inverted permutation tensor.
        """
        del name
        return _wrap_tensor(_k_ops.argsort(_to_tensor(x)))

    @staticmethod
    def l2_normalize(
        x: TensorCompatible | TensorOperand = 0.0,
        axis: int | Sequence[int] | Tensor | TensorOperand = None,
        epsilon: float = 1e-12,
        name: str | None = None,
        dim: int | Sequence[int] | Tensor | TensorOperand = None,
    ) -> Tensor:
        """Normalize `x` along dimension `axis` using an L2 norm.

        Args:
            x: Input tensor.
            axis: Dimension(s) along which to normalize.
            epsilon: Lower bound value for the norm.
            name: Optional operation name.
            dim: Deprecated alias for `axis`.

        Returns:
            Tensor: L2-normalized tensor.
        """
        del name
        t_x = _to_tensor(x)
        eff_ax = _normalize_axis(axis if axis is not None else dim)
        sq_sum = _to_tensor(_k_ops.sum(_ops.square(t_x), axis=eff_ax, keepdims=True))
        inv_norm = _ops.rsqrt(_ops.maximum(sq_sum, _to_tensor(float(epsilon), dtype=t_x.dtype)))
        return _wrap_tensor(_ops.multiply(t_x, inv_norm))

    @staticmethod
    def lgamma(
        x: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the natural logarithm of the absolute value of `Gamma(x)` elementwise.

        Args:
            x: Input floating-point tensor.
            name: Optional operation name.

        Returns:
            Tensor: Log-gamma tensor.
        """
        del name
        t_x = _to_tensor(1.0 if x is None else x)
        if _tracer.is_tracing:
            return _wrap_tensor(_ops.lgamma(t_x))
        return _apply_scalar_fn(t_x, "lgamma")

    @staticmethod
    def lbeta(
        x: TensorCompatible | TensorOperand = 1.0,
        name: str | None = None,
    ) -> Tensor:
        """Compute `ln(|Beta(x)|) = sum(lgamma(x), axis=-1) - lgamma(sum(x, axis=-1))`.

        Args:
            x: Rank `N >= 1` floating-point tensor.
            name: Optional operation name.

        Returns:
            Tensor: Rank `N - 1` tensor reduced along the last dimension.
        """
        del name
        t_x = _to_tensor(1.0 if x is None else x)
        if _tracer.is_tracing:
            return _wrap_tensor(_ops.lbeta(t_x))
        if not t_x.shape:
            return Tensor(0.0, dtype=to_dtype(t_x.dtype))
        lg_x = _to_tensor(math.lgamma(t_x))
        sum_lg = _to_tensor(_k_ops.sum(lg_x, axis=-1))
        sum_x = _wrap_tensor(_k_ops.sum(t_x, axis=-1))
        lg_sum = _to_tensor(math.lgamma(sum_x))
        return _wrap_tensor(_ops.subtract(sum_lg, lg_sum))

    @staticmethod
    def log_softmax(
        logits: TensorCompatible | TensorOperand = 0.0,
        axis: int | None = None,
        name: str | None = None,
    ) -> Tensor:
        """Compute log-softmax activations along `axis`.

        Args:
            logits: Non-empty floating-point tensor.
            axis: Dimension along which log-softmax is performed (default `-1`).
            name: Optional operation name.

        Returns:
            Tensor: Log-softmax tensor.
        """
        del name
        eff_ax = int(axis) if isinstance(axis, int) else -1
        return _wrap_tensor(_k_ops.log_softmax(_to_tensor(logits), axis=eff_ax))

    @staticmethod
    def softmax(
        logits: TensorCompatible | TensorOperand = 0.0,
        axis: int | None = None,
        name: str | None = None,
    ) -> Tensor:
        """Compute softmax activations along `axis`.

        Args:
            logits: Non-empty floating-point tensor.
            axis: Dimension along which softmax is performed (default `-1`).
            name: Optional operation name.

        Returns:
            Tensor: Softmax tensor.
        """
        del name
        eff_ax = int(axis) if isinstance(axis, int) else -1
        return _wrap_tensor(_k_ops.softmax(_to_tensor(logits), axis=eff_ax))

    @staticmethod
    def ndtri(
        x: TensorCompatible | TensorOperand = 0.5,
        name: str | None = None,
    ) -> Tensor:
        """Compute the inverse standard normal CDF `sqrt(2) * erfinv(2 * x - 1)` elementwise.

        Args:
            x: Input probability tensor in `(0, 1)`.
            name: Optional operation name.

        Returns:
            Tensor: Standard normal quantile tensor.
        """
        del name
        t_x = _to_tensor(0.5 if x is None else x)
        if _tracer.is_tracing:
            return _wrap_tensor(_ops.ndtri(t_x))
        two = _to_tensor(2.0, dtype=t_x.dtype)
        one = _to_tensor(1.0, dtype=t_x.dtype)
        sqrt2 = _to_tensor(_py_math.sqrt(2.0), dtype=t_x.dtype)
        return _wrap_tensor(
            _ops.multiply(sqrt2, _ops.erfinv(_ops.subtract(_ops.multiply(two, t_x), one)))
        )

    @staticmethod
    def polyval(
        coeffs: Sequence[TensorCompatible] | TensorCompatible | TensorOperand = (),
        x: TensorCompatible | TensorOperand = 0.0,
        name: str | None = None,
    ) -> Tensor:
        """Evaluate a polynomial with coefficients `coeffs` at `x` using Horner's method.

        Args:
            coeffs: Sequence of polynomial coefficients ordered from highest degree to constant term.
            x: Input tensor at which to evaluate the polynomial.
            name: Optional operation name.

        Returns:
            Tensor: Evaluated polynomial tensor matching the shape of `x`.
        """
        del name
        t_x = _to_tensor(0.0 if x is None else x)
        if isinstance(coeffs, (builtins.list, builtins.tuple)):
            c_list = [_to_tensor(c, dtype=t_x.dtype) for c in coeffs]
        elif isinstance(coeffs, Tensor) and len(coeffs.shape) == 1:
            n_c = coeffs.shape[0]
            c_list = [
                _to_tensor(_k_ops.take(_to_tensor(coeffs), i, axis=0)) for i in builtins.range(n_c)
            ]
        elif coeffs is None:
            c_list = []
        else:
            c_list = [_to_tensor(coeffs, dtype=t_x.dtype)]
        zero_t = _ops.multiply(t_x, _to_tensor(0.0, dtype=t_x.dtype))
        if not c_list:
            return _wrap_tensor(zero_t)
        acc = _ops.add(c_list[0], zero_t)
        for c_item in c_list[1:]:
            acc = _ops.add(_ops.multiply(acc, t_x), c_item)
        return _wrap_tensor(acc)

    @staticmethod
    def segment_max(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the maximum along segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: 1-D tensor of sorted non-negative segment indices.
            name: Optional operation name.

        Returns:
            Tensor: Segment maximum tensor of shape `(num_segments, *data.shape[1:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, None, "max", is_unsorted=False)

    @staticmethod
    def segment_mean(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the mean along segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: 1-D tensor of sorted non-negative segment indices.
            name: Optional operation name.

        Returns:
            Tensor: Segment mean tensor of shape `(num_segments, *data.shape[1:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, None, "mean", is_unsorted=False)

    @staticmethod
    def segment_min(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the minimum along segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: 1-D tensor of sorted non-negative segment indices.
            name: Optional operation name.

        Returns:
            Tensor: Segment minimum tensor of shape `(num_segments, *data.shape[1:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, None, "min", is_unsorted=False)

    @staticmethod
    def segment_prod(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the product along segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: 1-D tensor of sorted non-negative segment indices.
            name: Optional operation name.

        Returns:
            Tensor: Segment product tensor of shape `(num_segments, *data.shape[1:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, None, "prod", is_unsorted=False)

    @staticmethod
    def segment_sum(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the sum along segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: 1-D tensor of sorted non-negative segment indices.
            name: Optional operation name.

        Returns:
            Tensor: Segment sum tensor of shape `(num_segments, *data.shape[1:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, None, "sum", is_unsorted=False)

    @staticmethod
    def unsorted_segment_max(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        num_segments: int | TensorCompatible | TensorOperand = 1,
        name: str | None = None,
    ) -> Tensor:
        """Compute the maximum along unsorted segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: Tensor whose shape is a prefix of `data.shape`.
            num_segments: Total number of distinct segment IDs.
            name: Optional operation name.

        Returns:
            Tensor: Unsorted segment maximum tensor of shape `(num_segments, *data.shape[segment_ids.ndim:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, num_segments, "max", is_unsorted=True)

    @staticmethod
    def unsorted_segment_mean(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        num_segments: int | TensorCompatible | TensorOperand = 1,
        name: str | None = None,
    ) -> Tensor:
        """Compute the mean along unsorted segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: Tensor whose shape is a prefix of `data.shape`.
            num_segments: Total number of distinct segment IDs.
            name: Optional operation name.

        Returns:
            Tensor: Unsorted segment mean tensor of shape `(num_segments, *data.shape[segment_ids.ndim:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, num_segments, "mean", is_unsorted=True)

    @staticmethod
    def unsorted_segment_min(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        num_segments: int | TensorCompatible | TensorOperand = 1,
        name: str | None = None,
    ) -> Tensor:
        """Compute the minimum along unsorted segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: Tensor whose shape is a prefix of `data.shape`.
            num_segments: Total number of distinct segment IDs.
            name: Optional operation name.

        Returns:
            Tensor: Unsorted segment minimum tensor of shape `(num_segments, *data.shape[segment_ids.ndim:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, num_segments, "min", is_unsorted=True)

    @staticmethod
    def unsorted_segment_prod(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        num_segments: int | TensorCompatible | TensorOperand = 1,
        name: str | None = None,
    ) -> Tensor:
        """Compute the product along unsorted segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: Tensor whose shape is a prefix of `data.shape`.
            num_segments: Total number of distinct segment IDs.
            name: Optional operation name.

        Returns:
            Tensor: Unsorted segment product tensor of shape `(num_segments, *data.shape[segment_ids.ndim:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, num_segments, "prod", is_unsorted=True)

    @staticmethod
    def unsorted_segment_sqrt_n(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        num_segments: int | TensorCompatible | TensorOperand = 1,
        name: str | None = None,
    ) -> Tensor:
        """Compute the sum divided by `sqrt(N)` along unsorted segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: Tensor whose shape is a prefix of `data.shape`.
            num_segments: Total number of distinct segment IDs.
            name: Optional operation name.

        Returns:
            Tensor: Unsorted segment `sqrt_n` tensor of shape `(num_segments, *data.shape[segment_ids.ndim:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, num_segments, "sqrt_n", is_unsorted=True)

    @staticmethod
    def unsorted_segment_sum(
        data: TensorCompatible | TensorOperand = 0.0,
        segment_ids: TensorCompatible | TensorOperand = 0,
        num_segments: int | TensorCompatible | TensorOperand = 1,
        name: str | None = None,
    ) -> Tensor:
        """Compute the sum along unsorted segments of a tensor.

        Args:
            data: N-D input tensor.
            segment_ids: Tensor whose shape is a prefix of `data.shape`.
            num_segments: Total number of distinct segment IDs.
            name: Optional operation name.

        Returns:
            Tensor: Unsorted segment sum tensor of shape `(num_segments, *data.shape[segment_ids.ndim:])`.
        """
        del name
        return _segment_reduce_nd(data, segment_ids, num_segments, "sum", is_unsorted=True)

    @staticmethod
    def sobol_sample(
        dim: int | TensorCompatible | TensorOperand = 1,
        num_results: int | TensorCompatible | TensorOperand = 1,
        skip: int = 0,
        dtype: DTypeLike | TensorOperand = DType.Float32,
        name: str | None = None,
    ) -> Tensor:
        """Generate points from a Sobol low-discrepancy sequence.

        Args:
            dim: Positive integer dimensionality of each sample.
            num_results: Positive integer number of Sobol points to generate.
            skip: Number of initial points of the Sobol sequence to skip.
            dtype: Output floating-point dtype (`float32` or `float64`).
            name: Optional operation name.

        Returns:
            Tensor: 2-D `[num_results, dim]` tensor of Sobol points.
        """
        del skip, dtype, name
        d_int = int(dim) if isinstance(dim, (int, float)) else 1
        n_int = int(num_results) if isinstance(num_results, (int, float)) else 1
        if _tracer.is_tracing:
            return _wrap_tensor(as_operand(_ops.sobol_sample(d_int, n_int)))
        return _wrap_tensor(_k_ops.zeros((n_int, d_int)))

    @staticmethod
    def top_k(
        input: TensorCompatible | TensorOperand = 0.0,
        k: int = 1,
        sorted: builtins.bool = True,
        index_type: DTypeLike | TensorOperand = DType.Int32,
        name: str | None = None,
    ) -> builtins.tuple[Tensor, Tensor]:
        """Find values and indices of the `k` largest entries along the last dimension.

        Args:
            input: 1-D or higher tensor with last dimension at least `k`.
            k: Number of top elements to look for along the last dimension.
            sorted: If True, the resulting `k` elements will be sorted in descending order.
            index_type: Dtype for output indices.
            name: Optional operation name.

        Returns:
            tuple[Tensor, Tensor]: `(values, indices)` tuple of tensors.
        """
        del index_type, name
        t_in = _to_tensor(input)
        v_k, i_k = _k_ops.top_k(t_in, k=int(k), sorted=builtins.bool(sorted))
        return _wrap_tensor(v_k), _wrap_tensor(i_k)


abs = math.abs
acos = math.acos
acosh = math.acosh
add = math.add
reduce_all = math.reduce_all
reduce_any = math.reduce_any


def range(
    start: TensorCompatible | TensorOperand = 0,
    limit: TensorCompatible | TensorOperand = None,
    delta: TensorCompatible | TensorOperand = 1,
    dtype: DTypeLike | TensorOperand = None,
    name: str | None = "range",
) -> Tensor:
    """Create a 1-D sequence of numbers from `start` to `limit` with step `delta`.

    Args:
        start: First entry in the range (or `limit` if `limit` is `None`).
        limit: Upper limit of sequence (exclusive).
        delta: Step increment between consecutive values.
        dtype: Optional dtype of the resulting tensor.
        name: Optional operation name.

    Returns:
        Tensor: 1-D tensor of evenly spaced values.
    """
    del name
    dt_val = to_dtype(dtype).value if dtype is not None else None
    s_val = 0 if start is None else start
    d_val = 1 if delta is None else delta
    if limit is None:
        return _wrap_tensor(_k_ops.arange(s_val, dtype=dt_val))
    return _wrap_tensor(_k_ops.arange(s_val, limit, d_val, dtype=dt_val))


argmax = math.argmax
argmin = math.argmin
asin = math.asin
asinh = math.asinh
atan = math.atan
atan2 = math.atan2
atanh = math.atanh


_BITCAST_FMT: dict[str, str] = {
    "float16": "<e",
    "bfloat16": "<e",
    "float32": "<f",
    "float64": "<d",
    "int8": "<b",
    "qint8": "<b",
    "uint8": "<B",
    "quint8": "<B",
    "bool": "<B",
    "int16": "<h",
    "qint16": "<h",
    "uint16": "<H",
    "quint16": "<H",
    "int32": "<i",
    "qint32": "<i",
    "uint32": "<I",
    "int64": "<q",
    "uint64": "<Q",
}


def bitcast(
    input: TensorCompatible | TensorOperand = 0.0,
    type: DTypeLike | TensorOperand = DType.Float32,
    name: str | None = None,
) -> Tensor:
    """Bitcast a tensor from one type to another without copying data.

    Args:
        input: Input tensor to bitcast.
        type: Target `DType`.
        name: Optional operation name.

    Returns:
        Tensor: Bitcasted tensor.

    Raises:
        ValueError: If bitcasting to a wider dtype when the innermost dimension does not match the size ratio.
    """
    del name
    t_in = _to_tensor(0.0 if input is None else input)
    dst_dt = to_dtype(DType.Float32 if type is None else type)
    if _tracer.is_tracing:
        return _wrap_tensor(_ops.bitcast(t_in, dtype=dst_dt.as_compiler_dtype))
    src_dt = to_dtype(t_in.dtype)
    src_fmt = _BITCAST_FMT.get(src_dt.name, "<f")
    dst_fmt = _BITCAST_FMT.get(dst_dt.name, "<f")
    src_size = struct.calcsize(src_fmt)
    dst_size = struct.calcsize(dst_fmt)
    flat_vals = _flatten_floats(_to_py_data(t_in))
    raw_bytes = b"".join(
        struct.pack(src_fmt, int(v) if src_dt.is_integer or src_dt.is_bool else float(v))
        for v in flat_vals
    )
    if src_size == dst_size:
        out_shape = t_in.shape
    elif src_size > dst_size:
        out_shape = (*t_in.shape, src_size // dst_size)
    else:
        ratio = dst_size // src_size
        if not t_in.shape or t_in.shape[-1] != ratio:
            raise ValueError(
                f"Cannot bitcast from {src_dt.name} of shape {t_in.shape} to {dst_dt.name}"
            )
        out_shape = t_in.shape[:-1]
    n_dst = len(raw_bytes) // dst_size
    unpacked = [
        struct.unpack_from(dst_fmt, raw_bytes, idx * dst_size)[0] for idx in builtins.range(n_dst)
    ]
    if not out_shape:
        return Tensor(unpacked[0] if unpacked else 0, dtype=dst_dt)
    return _wrap_tensor(_k_ops.reshape(_to_tensor(unpacked, dtype=dst_dt), out_shape))


def broadcast_to(
    input: TensorCompatible | TensorOperand = 0.0,
    shape: ShapeLike | TensorOperand = (),
    name: str | None = None,
) -> Tensor:
    """Broadcast a tensor to a compatible target `shape`.

    Args:
        input: Input tensor.
        shape: Target broadcast shape.
        name: Optional operation name.

    Returns:
        Tensor: Broadcasted tensor.
    """
    del name
    return _wrap_tensor(_k_ops.broadcast_to(_to_tensor(input), shape))


def cast(
    x: TensorCompatible | TensorOperand = 0.0,
    dtype: DTypeLike | TensorOperand = DType.Float32,
    name: str | None = None,
) -> Tensor:
    """Cast a tensor `x` to a new `dtype`.

    Args:
        x: Input tensor.
        dtype: Destination dtype.
        name: Optional operation name.

    Returns:
        Tensor: Casted tensor.
    """
    del name
    dt_str = to_dtype(DType.Float32 if dtype is None else dtype).value
    return _wrap_tensor(_k_ops.cast(_to_tensor(x), dt_str))


def concat(
    values: Sequence[TensorCompatible] | TensorCompatible | TensorOperand = (),
    axis: int | Tensor | TensorOperand = 0,
    name: str | None = "concat",
) -> Tensor:
    """Concatenate tensors along one dimension `axis`.

    Args:
        values: Sequence of tensor objects or single tensor.
        axis: 0-D `int32` dimension along which to concatenate.
        name: Optional operation name.

    Returns:
        Tensor: Concatenated tensor.
    """
    del name
    first = (
        [_to_tensor(item) for item in values]
        if isinstance(values, (builtins.list, builtins.tuple)) and len(values) > 0
        else [
            _to_tensor(
                0.0
                if values is None or isinstance(values, (builtins.list, builtins.tuple))
                else values
            )
        ]
    )
    ax = _normalize_axis(axis)
    eff_ax = ax if isinstance(ax, int) else 0
    return _wrap_tensor(_k_ops.concatenate(first, axis=eff_ax))


cos = math.cos
cosh = math.cosh
divide = math.divide


def einsum(
    equation: str | TensorOperand = "",
    *inputs: TensorCompatible | TensorOperand,
    **kwargs: str | DTypeLike | TensorOperand,
) -> Tensor:
    """Tensor contraction over specified indices and outer product (`tf.einsum`).

    Args:
        equation: Einstein summation specification string.
        *inputs: Input tensors to contract.
        **kwargs: Optional keyword arguments (`name`, `optimize`).

    Returns:
        Tensor: Contracted output tensor.
    """
    del kwargs
    eq_str = str(equation) if equation is not None else ""
    t_inputs = [_to_tensor(a) for a in inputs]
    return _wrap_tensor(_k_ops.einsum(eq_str, *t_inputs))


equal = math.equal
exp = math.exp


def expand_dims(
    input: TensorCompatible | TensorOperand = 0.0,
    axis: int | Tensor | TensorOperand = 0,
    name: str | None = None,
) -> Tensor:
    """Insert a dimension of length `1` at dimension index `axis` of `input`'s shape.

    Args:
        input: Input tensor.
        axis: Dimension index at which to expand the shape of `input`.
        name: Optional operation name.

    Returns:
        Tensor: Tensor with an additional dimension of size `1`.
    """
    del name
    ax = _normalize_axis(axis)
    eff_ax = ax if isinstance(ax, int) else 0
    return _wrap_tensor(_k_ops.expand_dims(_to_tensor(input), eff_ax))


def eye(
    num_rows: int | TensorCompatible | TensorOperand = 1,
    num_columns: int | TensorCompatible | TensorOperand = None,
    batch_shape: ShapeLike | TensorOperand = None,
    dtype: DTypeLike | TensorOperand = DType.Float32,
    name: str | None = "eye",
) -> Tensor:
    """Construct an identity matrix, or a batch of matrices.

    Args:
        num_rows: Non-negative `int32` scalar number of rows in each matrix.
        num_columns: Optional number of columns in each matrix (defaults to `num_rows`).
        batch_shape: Optional list/tuple of leading batch dimensions.
        dtype: Data type of the resulting matrix.
        name: Optional operation name.

    Returns:
        Tensor: 2-D or batched identity matrix tensor.
    """
    del batch_shape, name
    r = int(num_rows) if isinstance(num_rows, (int, float)) else 1
    c = int(num_columns) if isinstance(num_columns, (int, float)) else None
    dt_str = to_dtype(DType.Float32 if dtype is None else dtype).value
    return _wrap_tensor(_k_ops.eye(r, c, dtype=dt_str))


def reshape(
    tensor: TensorCompatible | TensorOperand = 0.0,
    shape: ShapeLike | TensorOperand = (),
    name: str | None = None,
) -> Tensor:
    """Reshape `tensor` to `shape`.

    Args:
        tensor: Input tensor.
        shape: Target shape sequence or tensor.
        name: Optional operation name.

    Returns:
        Tensor: Reshaped tensor.
    """
    del name
    return _wrap_tensor(_k_ops.reshape(_to_tensor(tensor), shape))


pow = math.pow
floor = math.floor


def fill(
    dims: ShapeLike | TensorOperand = (),
    value: TensorCompatible | TensorOperand = 0.0,
    name: str | None = None,
    layout: str | None = None,
) -> Tensor:
    """Create a tensor of shape `dims` filled with scalar `value`.

    Args:
        dims: 1-D sequence of non-negative integers representing the output shape.
        value: Scalar value to fill into the output tensor.
        name: Optional operation name.
        layout: Optional DTensor layout.

    Returns:
        Tensor: Filled tensor.
    """
    del name, layout
    return _wrap_tensor(_k_ops.full(dims, value))


def gather(
    params: TensorCompatible | TensorOperand = 0.0,
    indices: TensorCompatible | TensorOperand = 0,
    validate_indices: builtins.bool | None = None,
    axis: int | Tensor | TensorOperand = None,
    batch_dims: int = 0,
    name: str | None = None,
) -> Tensor:
    """Gather slices from `params` axis `axis` according to `indices`.

    Args:
        params: Tensor from which to gather values.
        indices: Index tensor of integers.
        validate_indices: Deprecated argument (ignored).
        axis: Axis in `params` to gather from.
        batch_dims: Number of leading batch dimensions.
        name: Optional operation name.

    Returns:
        Tensor: Gathered tensor.
    """
    del validate_indices, name
    t_params = _to_tensor(params)
    t_ind = _to_tensor(0 if indices is None else indices)
    b_dims = int(batch_dims)
    if b_dims < 0:
        b_dims += len(t_ind.shape)
    ax = _normalize_axis(axis)
    eff_axis = ax if isinstance(ax, int) else b_dims
    if b_dims > 0 and not _tracer.is_tracing and len(t_params.shape) > b_dims:
        if eff_axis < 0:
            eff_axis += len(t_params.shape)
        batch_shape = t_params.shape[:b_dims]
        batch_count = int(_py_math.prod(batch_shape))
        p_flat = _to_tensor(_k_ops.reshape(t_params, (batch_count, *t_params.shape[b_dims:])))
        i_flat = _to_tensor(_k_ops.reshape(t_ind, (batch_count, *t_ind.shape[b_dims:])))
        gathered_batches = [
            _to_tensor(
                _k_ops.take(
                    _to_tensor(_k_ops.take(p_flat, b_i, axis=0)),
                    _to_tensor(_k_ops.take(i_flat, b_i, axis=0)),
                    axis=eff_axis - b_dims,
                )
            )
            for b_i in builtins.range(batch_count)
        ]
        stacked = _to_tensor(_k_ops.stack(gathered_batches, axis=0))
        return _wrap_tensor(_k_ops.reshape(stacked, (*batch_shape, *stacked.shape[1:])))
    return _wrap_tensor(_k_ops.take(t_params, t_ind, axis=ax if isinstance(ax, int) else None))


def gather_nd(
    params: TensorCompatible | TensorOperand = 0.0,
    indices: TensorCompatible | TensorOperand = 0,
    batch_dims: int = 0,
    name: str | None = None,
    bad_indices_policy: str = "",
) -> Tensor:
    """Gather slices from `params` into a Tensor with shape specified by `indices`.

    Args:
        params: Tensor from which to gather values.
        indices: Index tensor with innermost dimension indexing `params`.
        batch_dims: Number of leading batch dimensions.
        name: Optional operation name.
        bad_indices_policy: Policy for handling out-of-bounds indices.

    Returns:
        Tensor: Gathered tensor.
    """
    del name, bad_indices_policy
    t_params = _to_tensor(params)
    t_ind = _to_tensor([[0]] if indices is None else indices)
    if _tracer.is_tracing or not t_params.shape or not t_ind.shape:
        return _wrap_tensor(_k_ops.take(t_params, t_ind))
    b_dims = int(batch_dims)
    if b_dims < 0:
        b_dims += builtins.max(0, len(t_ind.shape) - 1)
    batch_shape = t_params.shape[:b_dims]
    batch_count = int(_py_math.prod(batch_shape)) if batch_shape else 1
    sub_shape = t_params.shape[b_dims:]
    m_depth = builtins.min(len(sub_shape), builtins.max(1, t_ind.shape[-1]))
    lead_dims = sub_shape[:m_depth]
    slice_shape = sub_shape[m_depth:]
    lead_prod = int(_py_math.prod(lead_dims)) if lead_dims else 1
    p_reshaped = _to_tensor(_k_ops.reshape(t_params, (batch_count, lead_prod, *slice_shape)))
    ind_outer_shape = t_ind.shape[b_dims:-1]
    coords_per_batch = int(_py_math.prod(ind_outer_shape)) if ind_outer_shape else 1
    idx_flat = _flatten_ints(_to_py_data(t_ind))
    batch_outputs: builtins.list[Tensor] = []
    for b_i in builtins.range(batch_count):
        p_b = _to_tensor(_k_ops.take(p_reshaped, b_i, axis=0))
        lin_indices: builtins.list[int] = []
        for c_i in builtins.range(coords_per_batch):
            base_k = (b_i * coords_per_batch + c_i) * m_depth
            coord = idx_flat[base_k : base_k + m_depth]
            lin_idx = 0
            for c_val, d_len in zip(coord, lead_dims):
                lin_idx = lin_idx * d_len + builtins.max(0, builtins.min(d_len - 1, c_val))
            lin_indices.append(lin_idx)
        gathered_b = _to_tensor(
            _k_ops.take(p_b, _to_tensor(lin_indices, dtype=DType.Int32), axis=0)
        )
        batch_outputs.append(
            _wrap_tensor(_k_ops.reshape(gathered_b, (*ind_outer_shape, *slice_shape)))
        )
    if b_dims == 0:
        return batch_outputs[0]
    stacked_nd = _to_tensor(_k_ops.stack([_to_tensor(o) for o in batch_outputs], axis=0))
    return _wrap_tensor(_k_ops.reshape(stacked_nd, (*batch_shape, *ind_outer_shape, *slice_shape)))


greater = math.greater
greater_equal = math.greater_equal


def identity(
    input: TensorCompatible | TensorOperand = 0.0,
    name: str | None = None,
) -> Tensor:
    """Return a tensor with the same shape and contents as `input`.

    Args:
        input: Input tensor.
        name: Optional operation name.

    Returns:
        Tensor: Identity copy of `input`.
    """
    del name
    return _wrap_tensor(_to_tensor(0.0 if input is None else input))


less = math.less
less_equal = math.less_equal


def linspace(
    start: TensorCompatible | TensorOperand = 0.0,
    stop: TensorCompatible | TensorOperand = 1.0,
    num: int | TensorCompatible | TensorOperand = 1,
    name: str | None = None,
    axis: int = 0,
) -> Tensor:
    """Generate `num` evenly-spaced values in an interval `[start, stop]` along `axis`.

    Args:
        start: Starting value(s) of the range.
        stop: Ending value(s) of the range.
        num: Number of values to generate.
        name: Optional operation name.
        axis: Axis in the result to store the samples.

    Returns:
        Tensor: Evenly spaced values tensor.
    """
    del name, axis
    return _wrap_tensor(_k_ops.linspace(start, stop, num))


logical_and = math.logical_and
logical_not = math.logical_not
logical_or = math.logical_or


def matmul(
    a: TensorCompatible | TensorOperand = 0.0,
    b: TensorCompatible | TensorOperand = 0.0,
    transpose_a: builtins.bool = False,
    transpose_b: builtins.bool = False,
    adjoint_a: builtins.bool = False,
    adjoint_b: builtins.bool = False,
    a_is_sparse: builtins.bool = False,
    b_is_sparse: builtins.bool = False,
    output_type: DTypeLike | TensorOperand = None,
    grad_a: builtins.bool = False,
    grad_b: builtins.bool = False,
    name: str | None = None,
) -> Tensor:
    """Multiply matrix `a` by matrix `b`, producing `a @ b`.

    Args:
        a: First matrix tensor of rank `>= 2`.
        b: Second matrix tensor of matching rank `>= 2`.
        transpose_a: If True, `a` is transposed before multiplication.
        transpose_b: If True, `b` is transposed before multiplication.
        adjoint_a: If True, `a` is conjugated and transposed before multiplication.
        adjoint_b: If True, `b` is conjugated and transposed before multiplication.
        a_is_sparse: Hint whether `a` is sparse.
        b_is_sparse: Hint whether `b` is sparse.
        output_type: Optional output dtype.
        grad_a: Hint whether gradient of `a` is computed.
        grad_b: Hint whether gradient of `b` is computed.
        name: Optional operation name.

    Returns:
        Tensor: Matrix product tensor.
    """
    del a_is_sparse, b_is_sparse, output_type, grad_a, grad_b, name
    t_a = _to_tensor(a)
    t_b = _to_tensor(b)
    if (transpose_a or adjoint_a) and len(t_a.shape) >= 2:
        rk_a = len(t_a.shape)
        perm_a = (*builtins.range(rk_a - 2), rk_a - 1, rk_a - 2)
        t_a = _to_tensor(_k_ops.transpose(t_a, perm_a))
        if adjoint_a:
            t_a = _ops.conj(t_a)
    if (transpose_b or adjoint_b) and len(t_b.shape) >= 2:
        rk_b = len(t_b.shape)
        perm_b = (*builtins.range(rk_b - 2), rk_b - 1, rk_b - 2)
        t_b = _to_tensor(_k_ops.transpose(t_b, perm_b))
        if adjoint_b:
            t_b = _ops.conj(t_b)
    return _wrap_tensor(_k_ops.matmul(t_a, t_b))


reduce_max = math.reduce_max
maximum = math.maximum
reduce_mean = math.reduce_mean


def meshgrid(
    *args: TensorCompatible | TensorOperand,
    **kwargs: str | TensorOperand,
) -> builtins.list[Tensor]:
    """Broadcast parameters for evaluation on an N-D coordinate grid.

    Args:
        *args: 1-D coordinate tensors.
        **kwargs: Keyword options (`indexing='xy' | 'ij'`, `name`).

    Returns:
        list[Tensor]: List of N-D coordinate grid tensors.

    Raises:
        ValueError: If `indexing` is not `'xy'` or `'ij'`.
    """
    tensors = [_to_tensor(a) for a in args if a is not None]
    if not tensors:
        return []
    indexing_raw = kwargs.get("indexing", "xy")
    indexing_mode = str(indexing_raw).lower() if indexing_raw is not None else "xy"
    if indexing_mode not in ("xy", "ij"):
        raise ValueError(f"indexing parameter must be either 'xy' or 'ij', got {indexing_mode!r}")
    lengths = [int(_py_math.prod(t.shape)) if t.shape else 1 for t in tensors]
    flat_tensors = [_to_tensor(_k_ops.reshape(t, (lengths[i],))) for i, t in enumerate(tensors)]
    if len(flat_tensors) == 1:
        return [_wrap_tensor(flat_tensors[0])]
    n_dims = len(flat_tensors)
    grid_shape = (
        (lengths[1], lengths[0], *lengths[2:]) if indexing_mode == "xy" else builtins.tuple(lengths)
    )
    grids: builtins.list[Tensor] = []
    for i, t_1d in enumerate(flat_tensors):
        target_axis = (1 if i == 0 else (0 if i == 1 else i)) if indexing_mode == "xy" else i
        view_shape = builtins.tuple(
            lengths[i] if ax == target_axis else 1 for ax in builtins.range(n_dims)
        )
        reshaped = _to_tensor(_k_ops.reshape(t_1d, view_shape))
        grids.append(_wrap_tensor(_k_ops.broadcast_to(reshaped, grid_shape)))
    return grids


reduce_min = math.reduce_min
minimum = math.minimum
multiply = math.multiply
negative = math.negative


def norm(
    tensor: TensorCompatible | TensorOperand = 0.0,
    ord: int | float | str | TensorOperand = "euclidean",
    axis: int | Sequence[int] | Tensor | TensorOperand = None,
    keepdims: builtins.bool | None = None,
    name: str | None = None,
) -> Tensor:
    """Compute the norm of vectors, matrices, and tensors.

    Args:
        tensor: Input tensor.
        ord: Order of the norm (`'euclidean'`, `'fro'`, `1`, `2`, `inf`, `-inf`, or real `p`).
        axis: Dimension(s) along which to compute the norm.
        keepdims: If True, retain reduced dimensions with size `1`.
        name: Optional operation name.

    Returns:
        Tensor: Computed norm tensor.
    """
    del name
    t_in = _to_tensor(0.0 if tensor is None else tensor)
    axis_val = _normalize_axis(axis)
    kd_val = builtins.bool(keepdims) if keepdims is not None else False
    if ord in (None, "euclidean", "fro", 2, 2.0):
        res = _ops.sqrt(_to_tensor(_k_ops.sum(_ops.square(t_in), axis=axis_val, keepdims=kd_val)))
    elif ord in (1, 1.0):
        res = _to_tensor(_k_ops.sum(_ops.abs(t_in), axis=axis_val, keepdims=kd_val))
    elif ord == float("inf") or ord == "inf":
        res = _to_tensor(_k_ops.max(_ops.abs(t_in), axis=axis_val, keepdims=kd_val))
    elif ord == float("-inf") or ord == "-inf":
        res = _to_tensor(_k_ops.min(_ops.abs(t_in), axis=axis_val, keepdims=kd_val))
    elif isinstance(ord, (int, float)) and not isinstance(ord, builtins.bool):
        p_f = float(ord)
        p_t = _to_tensor(p_f, dtype=t_in.dtype)
        inv_p_t = _to_tensor(1.0 / p_f, dtype=t_in.dtype)
        sum_p = _to_tensor(
            _k_ops.sum(_ops.power(_ops.abs(t_in), p_t), axis=axis_val, keepdims=kd_val)
        )
        res = _ops.power(sum_p, inv_p_t)
    else:
        res = _ops.norm(t_in, ord=str(ord), axis=axis_val, keepdims=kd_val)
    return _wrap_tensor(res)


not_equal = math.not_equal


def ones(
    shape: ShapeLike | TensorOperand = (),
    dtype: DTypeLike | TensorOperand = DType.Float32,
    name: str | None = None,
    layout: str | None = None,
) -> Tensor:
    """Create a tensor with all elements set to `1`.

    Args:
        shape: Shape of the output tensor.
        dtype: Data type of the output tensor.
        name: Optional operation name.
        layout: Optional DTensor layout.

    Returns:
        Tensor: All-ones tensor.
    """
    del name, layout
    dt_str = to_dtype(DType.Float32 if dtype is None else dtype).value
    return _wrap_tensor(_k_ops.ones(shape, dtype=dt_str))


def ones_like(
    input: TensorCompatible | TensorOperand = 0.0,
    dtype: DTypeLike | TensorOperand = None,
    name: str | None = None,
    layout: str | None = None,
) -> Tensor:
    """Create a tensor of all ones with the same shape as `input`.

    Args:
        input: Input tensor whose shape is copied.
        dtype: Optional override dtype.
        name: Optional operation name.
        layout: Optional DTensor layout.

    Returns:
        Tensor: All-ones tensor matching `input`'s shape.
    """
    del name, layout
    dt_str = to_dtype(dtype).value if dtype is not None else None
    return _wrap_tensor(_k_ops.ones_like(_to_tensor(input), dtype=dt_str))


reduce_prod = math.reduce_prod


def repeat(
    input: TensorCompatible | TensorOperand = 0.0,
    repeats: int | Sequence[int] | TensorCompatible | TensorOperand = 1,
    axis: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Repeat elements of `input`.

    Args:
        input: Input tensor.
        repeats: Number of repetitions for each element.
        axis: Axis along which to repeat values.
        name: Optional operation name.

    Returns:
        Tensor: Tensor with repeated elements.
    """
    del name
    return _wrap_tensor(_k_ops.repeat(_to_tensor(input), repeats, axis=axis))


def roll(
    input: TensorCompatible | TensorOperand = 0.0,
    shift: int | Sequence[int] | TensorCompatible | TensorOperand = 0,
    axis: int | Sequence[int] | Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor:
    """Roll the elements of a tensor along an axis.

    Args:
        input: Input tensor.
        shift: Offset(s) by which elements are shifted.
        axis: Axis or axes along which elements are shifted.
        name: Optional operation name.

    Returns:
        Tensor: Shifted tensor.
    """
    del name
    ax = _normalize_axis(axis)
    return _wrap_tensor(_k_ops.roll(_to_tensor(input), shift, axis=ax))


round = math.round


def scatter_nd(
    indices: TensorCompatible | TensorOperand = 0,
    updates: TensorCompatible | TensorOperand = 0.0,
    shape: ShapeLike | TensorOperand = None,
    bad_indices_policy: str = "",
    name: str | None = None,
) -> Tensor:
    """Scatter `updates` into a new zero-initialized tensor of shape `shape` according to `indices`.

    Args:
        indices: Index tensor.
        updates: Values to scatter into the output tensor.
        shape: Shape of the resulting tensor.
        bad_indices_policy: Policy for handling out-of-bounds indices.
        name: Optional operation name.

    Returns:
        Tensor: Scattered tensor of shape `shape`.
    """
    del bad_indices_policy, name
    updates_t = _to_tensor(0.0 if updates is None else updates)
    shape_arg = updates_t.shape if shape is None else shape
    base = _k_ops.zeros(shape_arg, dtype=updates_t.dtype)
    idx_t = _to_tensor([0] if indices is None else indices)
    if _tracer.is_tracing:
        return _wrap_tensor(_k_ops.scatter(base, 0, idx_t, updates_t))
    scattered = tensor_scatter_nd_add(
        _wrap_tensor(base), _wrap_tensor(idx_t), _wrap_tensor(updates_t)
    )
    assert scattered is not None
    return scattered


def shape(
    input: TensorCompatible | TensorOperand = 0.0,
    out_type: DTypeLike | TensorOperand = DType.Int32,
    name: str | None = None,
) -> builtins.tuple[Tensor, ...]:
    """Return the shape of a tensor as a tuple of scalar `Tensor` dimensions.

    Args:
        input: Input tensor.
        out_type: Output dimension dtype (`int32` or `int64`).
        name: Optional operation name.

    Returns:
        tuple[Tensor, ...]: Tuple of dimension `Tensor`s.
    """
    del out_type, name
    res = _k_ops.shape(_to_tensor(input))
    wrapped = _wrap(res)
    return (
        wrapped
        if isinstance(wrapped, builtins.tuple)
        else (wrapped if isinstance(wrapped, Tensor) else _wrap_tensor(wrapped),)
    )


sign = math.sign
sin = math.sin
sinh = math.sinh


def slice(
    input_: TensorCompatible | TensorOperand = 0.0,
    begin: Sequence[int] | int | TensorCompatible | TensorOperand = (),
    size: Sequence[int] | int | TensorCompatible | TensorOperand = (),
    name: str | None = None,
) -> Tensor:
    """Extract a slice of size `size` from `input_` starting at location `begin`.

    Args:
        input_: Input tensor.
        begin: Starting location for each dimension of `input_`.
        size: Number of elements to slice along each dimension (`-1` for all remaining).
        name: Optional operation name.

    Returns:
        Tensor: Sliced tensor.
    """
    del name
    curr = _to_tensor(input_)
    b_seq = (
        builtins.list(begin)
        if isinstance(begin, (builtins.list, builtins.tuple))
        else (
            [int(begin)]
            if isinstance(begin, int)
            else _flatten_ints(_to_py_data(_to_tensor(begin)))
        )
    )
    sz_seq = (
        builtins.list(size)
        if isinstance(size, (builtins.list, builtins.tuple))
        else (
            [int(size)] if isinstance(size, int) else _flatten_ints(_to_py_data(_to_tensor(size)))
        )
    )
    for axis, (b, sz) in enumerate(zip(b_seq, sz_seq)):
        if axis >= len(curr.shape):
            break
        b_int = int(b) if isinstance(b, int) else 0
        sz_int = int(sz) if isinstance(sz, int) else 0
        end_int = None if sz_int < 0 else b_int + sz_int
        curr = _to_tensor(_k_ops.slice(curr, axis, b_int, end_int, 1))
    return _wrap_tensor(curr)


def split(
    value: TensorCompatible | TensorOperand = 0.0,
    num_or_size_splits: int | Sequence[int] | TensorCompatible | TensorOperand = 1,
    axis: int = 0,
    num: int | None = None,
    name: str | None = "split",
) -> builtins.list[Tensor]:
    """Split a tensor `value` into a list of sub-tensors along `axis`.

    Args:
        value: Input tensor to split.
        num_or_size_splits: Either an integer number of equal splits or a 1-D sequence of split sizes.
        axis: Dimension along which to split.
        num: Optional number of outputs when `size_splits` cannot be statically inferred.
        name: Optional operation name.

    Returns:
        list[Tensor]: List of split sub-tensors.
    """
    del num, name
    t_val = _to_tensor(0.0 if value is None else value)
    eff_axis = int(axis)
    if eff_axis < 0 and t_val.shape:
        eff_axis += len(t_val.shape)
    size_splits: builtins.list[int] | None = None
    if isinstance(num_or_size_splits, (builtins.list, builtins.tuple)):
        size_splits = [int(s) for s in num_or_size_splits if isinstance(s, int)]
    elif isinstance(num_or_size_splits, Tensor) and len(num_or_size_splits.shape) == 1:
        raw_list = num_or_size_splits.tolist()
        assert isinstance(raw_list, builtins.list)
        size_splits = [int(s) for s in raw_list if isinstance(s, (int, float))]
    if size_splits is not None and 0 <= eff_axis < len(t_val.shape):
        dim_len = t_val.shape[eff_axis]
        known_sum = builtins.sum(s for s in size_splits if s >= 0)
        resolved = [dim_len - known_sum if s < 0 else s for s in size_splits]
        out_parts: builtins.list[Tensor] = []
        cur_start = 0
        for sz in resolved:
            out_parts.append(
                _wrap_tensor(_k_ops.slice(t_val, eff_axis, cur_start, cur_start + sz, 1))
            )
            cur_start += sz
        return out_parts
    n_splits = int(num_or_size_splits) if isinstance(num_or_size_splits, (int, float)) else 1
    res = _k_ops.split(t_val, n_splits, axis=eff_axis)
    wrapped = _wrap(res)
    return (
        builtins.list(wrapped)
        if isinstance(wrapped, (builtins.list, builtins.tuple))
        else [wrapped]
    )


sqrt = math.sqrt
square = math.square


def squeeze(
    input: TensorCompatible | TensorOperand = 0.0,
    axis: int | Sequence[int] | Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor:
    """Remove dimensions of size `1` from the shape of a tensor.

    Args:
        input: Input tensor.
        axis: Optional specific dimension index or indices of size `1` to remove.
        name: Optional operation name.

    Returns:
        Tensor: Squeezed tensor.

    Raises:
        ValueError: If an explicitly specified `axis` has dimension size `!= 1`.
    """
    del name
    t_in = _to_tensor(input)
    ax_norm = _normalize_axis(axis)
    axes_list = (
        [ax_norm]
        if isinstance(ax_norm, int)
        else (builtins.list(ax_norm) if isinstance(ax_norm, builtins.tuple) else [])
    )
    for ax_item in axes_list:
        norm_ax = ax_item if ax_item >= 0 else len(t_in.shape) + ax_item
        if norm_ax < 0 or norm_ax >= len(t_in.shape) or t_in.shape[norm_ax] != 1:
            raise ValueError(
                f"Can not squeeze dim[{ax_item}], expected a dimension of 1, "
                f"got {t_in.shape[norm_ax] if 0 <= norm_ax < len(t_in.shape) else 'out of range'}"
            )
    return _wrap_tensor(_k_ops.squeeze(t_in, axis=ax_norm))


def stack(
    values: Sequence[TensorCompatible] | TensorCompatible | TensorOperand = (),
    axis: int | Tensor | TensorOperand = 0,
    name: str | None = "stack",
) -> Tensor:
    """Stack a list of rank-`R` tensors into one rank-`(R+1)` tensor along `axis`.

    Args:
        values: Sequence of tensors with the same shape and dtype.
        axis: Axis along which to stack.
        name: Optional operation name.

    Returns:
        Tensor: Stacked tensor.
    """
    del name
    first = (
        [_to_tensor(item) for item in values]
        if isinstance(values, (builtins.list, builtins.tuple)) and len(values) > 0
        else [
            _to_tensor(
                0.0
                if values is None or isinstance(values, (builtins.list, builtins.tuple))
                else values
            )
        ]
    )
    ax = _normalize_axis(axis)
    eff_ax = ax if isinstance(ax, int) else 0
    return _wrap_tensor(_k_ops.stack(first, axis=eff_ax))


def strided_slice(
    input_: TensorCompatible | TensorOperand = 0.0,
    begin: Sequence[int] | TensorCompatible | TensorOperand = (),
    end: Sequence[int] | TensorCompatible | TensorOperand = (),
    strides: Sequence[int] | TensorCompatible | TensorOperand = None,
    begin_mask: int = 0,
    end_mask: int = 0,
    ellipsis_mask: int = 0,
    new_axis_mask: int = 0,
    shrink_axis_mask: int = 0,
    var: TensorCompatible | TensorOperand = None,
    name: str | None = None,
) -> Tensor:
    """Extract a strided slice of a tensor supporting bitmasks for begin/end/ellipsis/newaxis/shrink.

    Args:
        input_: Input tensor to slice.
        begin: 1-D sequence of start indices per slice specification.
        end: 1-D sequence of end indices per slice specification.
        strides: Optional 1-D sequence of step sizes per slice specification.
        begin_mask: Bitmask indicating dimensions whose start index should be ignored.
        end_mask: Bitmask indicating dimensions whose end index should be ignored.
        ellipsis_mask: Bitmask indicating an ellipsis (`...`) position.
        new_axis_mask: Bitmask indicating positions where a new length-1 axis is inserted.
        shrink_axis_mask: Bitmask indicating dimensions to shrink (squeeze) to a scalar index.
        var: Optional variable reference (unused).
        name: Optional operation name.

    Returns:
        Tensor: Strided-sliced tensor.
    """
    del var, name
    if (
        isinstance(begin, (builtins.list, builtins.tuple))
        and isinstance(end, (builtins.list, builtins.tuple))
        and len(begin) > 0
    ):
        curr = _to_tensor(input_)
        b_seq = builtins.list(begin)
        e_seq = builtins.list(end)
        strides_seq = (
            builtins.list(strides)
            if isinstance(strides, (builtins.list, builtins.tuple))
            else [1] * len(b_seq)
        )
        bm = int(begin_mask)
        em = int(end_mask)
        elm = int(ellipsis_mask)
        nam = int(new_axis_mask)
        sam = int(shrink_axis_mask)

        tensor_axis = 0
        out_axis = 0
        num_specs = len(b_seq)
        num_new_axes = builtins.sum(1 for i in builtins.range(num_specs) if (nam >> i) & 1)
        num_ellipses = builtins.sum(1 for i in builtins.range(num_specs) if (elm >> i) & 1)
        non_new_specs = num_specs - num_new_axes - num_ellipses
        ellipsis_span = builtins.max(0, len(curr.shape) - non_new_specs)

        for spec_idx in builtins.range(num_specs):
            if (elm >> spec_idx) & 1:
                tensor_axis += ellipsis_span
                out_axis += ellipsis_span
                continue
            if (nam >> spec_idx) & 1:
                curr = _to_tensor(_k_ops.expand_dims(curr, out_axis))
                out_axis += 1
                continue
            if out_axis >= len(curr.shape):
                break
            dim_size = curr.shape[out_axis]
            b_val = b_seq[spec_idx]
            e_val = e_seq[spec_idx]
            st_val = strides_seq[spec_idx] if spec_idx < len(strides_seq) else 1
            b_int = int(b_val) if isinstance(b_val, int) else 0
            e_int = int(e_val) if isinstance(e_val, int) else 0
            st_int = int(st_val) if isinstance(st_val, int) and int(st_val) != 0 else 1

            if (sam >> spec_idx) & 1:
                idx_norm = b_int if b_int >= 0 else dim_size + b_int
                curr = _to_tensor(_k_ops.slice(curr, out_axis, idx_norm, idx_norm + 1, 1))
                curr = _to_tensor(_k_ops.squeeze(curr, axis=out_axis))
                tensor_axis += 1
                continue

            eff_b = (0 if st_int > 0 else dim_size - 1) if ((bm >> spec_idx) & 1) else b_int
            eff_e = (dim_size if st_int > 0 else -dim_size - 1) if ((em >> spec_idx) & 1) else e_int
            curr = _to_tensor(_k_ops.slice(curr, out_axis, eff_b, eff_e, st_int))
            tensor_axis += 1
            out_axis += 1
        return _wrap_tensor(curr)
    return _wrap_tensor(_to_tensor(0.0 if input_ is None else input_))


subtract = math.subtract
reduce_sum = math.reduce_sum
tan = math.tan
tanh = math.tanh


def tile(
    input: TensorCompatible | TensorOperand = 0.0,
    multiples: Sequence[int] | TensorCompatible | TensorOperand = (),
    name: str | None = None,
) -> Tensor:
    """Construct a tensor by tiling a given tensor `input` by `multiples`.

    Args:
        input: Input tensor.
        multiples: 1-D sequence of replication factors per dimension.
        name: Optional operation name.

    Returns:
        Tensor: Tiled tensor.
    """
    del name
    return _wrap_tensor(_k_ops.tile(_to_tensor(input), multiples))


def transpose(
    a: TensorCompatible | TensorOperand = 0.0,
    perm: Sequence[int] | TensorCompatible | TensorOperand = None,
    conjugate: builtins.bool = False,
    name: str | None = "transpose",
) -> Tensor:
    """Transpose `a`, permuting dimensions according to `perm` and optionally conjugating.

    Args:
        a: Input tensor.
        perm: Permutation of the dimensions of `a`.
        conjugate: If True, compute the complex conjugate transpose.
        name: Optional operation name.

    Returns:
        Tensor: Transposed tensor.
    """
    del name
    perm_norm = _normalize_axis(perm)
    axes_arg = (perm_norm,) if isinstance(perm_norm, int) else perm_norm
    res = _k_ops.transpose(_to_tensor(a), axes=axes_arg)
    if conjugate:
        res = _ops.conj(_to_tensor(res))
    return _wrap_tensor(res)


def unstack(
    value: TensorCompatible | TensorOperand = 0.0,
    num: int | None = None,
    axis: int | Tensor | TensorOperand = 0,
    name: str | None = "unstack",
) -> builtins.list[Tensor]:
    """Unpack a dimension of a rank-`R` tensor into a list of rank-`(R-1)` tensors.

    Args:
        value: Rank `R > 0` tensor to unstack.
        num: Optional length of the dimension `axis`.
        axis: Axis along which to unstack.
        name: Optional operation name.

    Returns:
        list[Tensor]: List of unstacked tensors.
    """
    del num, name
    ax = _normalize_axis(axis)
    eff_ax = ax if isinstance(ax, int) else 0
    res = _k_ops.unstack(_to_tensor(value), axis=eff_ax)
    wrapped = _wrap(res)
    return (
        builtins.list(wrapped)
        if isinstance(wrapped, (builtins.list, builtins.tuple))
        else [wrapped]
    )


def _flatten_nonzero_flags(data: NestedTensorData) -> builtins.list[builtins.bool]:
    """Flatten nested Python lists/scalars into a list of truthy boolean flags.

    Args:
        data: Nested list or scalar data.

    Returns:
        list[bool]: Flattened truthiness flags.
    """
    if isinstance(data, (builtins.list, builtins.tuple)):
        out: builtins.list[builtins.bool] = []
        for item in data:
            out.extend(_flatten_nonzero_flags(item))
        return out
    return [builtins.bool(data)]


def where(
    condition: TensorCompatible | TensorOperand = True,
    x: TensorCompatible | TensorOperand = None,
    y: TensorCompatible | TensorOperand = None,
    name: str | None = None,
) -> Tensor:
    """Return the coordinates of true elements of `condition` (1-arg), or select elements from `x` or `y` (3-arg).

    Args:
        condition: Boolean or numeric condition tensor.
        x: Values selected at indices where `condition` is True.
        y: Values selected at indices where `condition` is False.
        name: Optional operation name.

    Returns:
        Tensor: 2-D `int64` coordinate tensor (when `x` and `y` are None) or selected tensor.
    """
    del name
    if x is None and y is None:
        t_cond = _to_tensor(condition)
        rank = len(t_cond.shape)
        if rank == 0:
            is_true = builtins.bool(_wrap_tensor(t_cond).tolist())
            return _wrap_tensor(
                _ops.zeros((1 if is_true else 0, 0), dtype=DType.Int64.as_compiler_dtype)
            )
        flat_flags = _flatten_nonzero_flags(_wrap_tensor(t_cond).tolist())
        all_coords = itertools.product(*(builtins.range(d) for d in t_cond.shape))
        true_coords = [builtins.list(coord) for coord, flag in zip(all_coords, flat_flags) if flag]
        if not true_coords:
            return _wrap_tensor(_ops.zeros((0, rank), dtype=DType.Int64.as_compiler_dtype))
        return Tensor(true_coords, dtype=DType.Int64)
    return _wrap_tensor(_k_ops.where(_to_tensor(condition), _to_tensor(x), _to_tensor(y)))


def zeros(
    shape: ShapeLike | TensorOperand = (),
    dtype: DTypeLike | TensorOperand = DType.Float32,
    name: str | None = None,
    layout: str | None = None,
) -> Tensor:
    """Create a tensor with all elements set to zero.

    Args:
        shape: Shape of the output tensor.
        dtype: Data type of the output tensor.
        name: Optional operation name.
        layout: Optional DTensor layout.

    Returns:
        Tensor: All-zeros tensor.
    """
    del name, layout
    dt_str = to_dtype(DType.Float32 if dtype is None else dtype).value
    return _wrap_tensor(_k_ops.zeros(shape, dtype=dt_str))


def zeros_like(
    input: TensorCompatible | TensorOperand = 0.0,
    dtype: DTypeLike | TensorOperand = None,
    name: str | None = None,
    layout: str | None = None,
) -> Tensor:
    """Create a tensor of all zeros with the same shape as `input`.

    Args:
        input: Input tensor whose shape is copied.
        dtype: Optional override dtype.
        name: Optional operation name.
        layout: Optional DTensor layout.

    Returns:
        Tensor: All-zeros tensor matching `input`'s shape.
    """
    del name, layout
    dt_str = to_dtype(dtype).value if dtype is not None else None
    return _wrap_tensor(_k_ops.zeros_like(_to_tensor(input), dtype=dt_str))


def tensordot(
    a: TensorCompatible | TensorOperand = 0.0,
    b: TensorCompatible | TensorOperand = 0.0,
    axes: int | Sequence[int] | Sequence[Sequence[int]] | TensorCompatible | TensorOperand = 1,
    name: str | None = None,
) -> Tensor:
    """Tensor contraction of `a` and `b` along specified `axes`.

    Args:
        a: First tensor operand.
        b: Second tensor operand.
        axes: Scalar integer or pair of axis sequences to contract.
        name: Optional operation name.

    Returns:
        Tensor: Contracted tensor.
    """
    del name
    return _wrap_tensor(_k_ops.tensordot(_to_tensor(a), _to_tensor(b), axes=axes))


scalar_mul = math.scalar_mul
truediv = math.truediv
