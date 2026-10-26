"""TensorFlow `math` namespace and top-level mathematical/array operations."""

from __future__ import annotations

import builtins

from zero_tensorflow._typing import TensorOperand, as_operand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap, _wrap_tensor

import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.core.dtype import DType
from zero_keras import ops as _k_ops

_IGNORED_KW: set[str] = {
    "a_is_sparse",
    "b_is_sparse",
    "bad_indices_policy",
    "batch_shape",
    "begin_mask",
    "conjugate",
    "ellipsis_mask",
    "end_mask",
    "grad_a",
    "grad_b",
    "index_type",
    "layout",
    "maxlength",
    "name",
    "new_axis_mask",
    "num",
    "out_type",
    "output_type",
    "shrink_axis_mask",
    "skip",
    "tensor_dtype",
    "validate_indices",
    "var",
}


def _pack_pos(
    pos_items: builtins.tuple[Tensor | TensorOperand, ...],
    extra_args: builtins.tuple[Tensor | TensorOperand, ...],
) -> builtins.tuple[Tensor | TensorOperand, ...]:
    """Pack explicit positional arguments and variadic `*extra_args`, omitting trailing `None`s.

    Args:
        pos_items: Tuple of explicit positional parameter values.
        extra_args: Variadic positional arguments.

    Returns:
        tuple[Tensor | TensorOperand, ...]: Combined positional argument tuple.
    """
    trimmed = builtins.list(pos_items)
    while trimmed and trimmed[-1] is None:
        trimmed.pop()
    return builtins.tuple(trimmed) + extra_args


def _pack_call(
    pos_items: builtins.tuple[Tensor | TensorOperand, ...],
    kw_items: builtins.tuple[builtins.tuple[str, Tensor | TensorOperand], ...],
    extra_args: builtins.tuple[Tensor | TensorOperand, ...],
    extra_kwargs: dict[str, Tensor | TensorOperand],
) -> builtins.tuple[
    builtins.tuple[Tensor | TensorOperand, ...],
    dict[str, Tensor | TensorOperand],
]:
    """Combine explicit positional and keyword parameters with `*extra_args` and `**extra_kwargs`.

    Args:
        pos_items: Tuple of explicit positional parameter values.
        kw_items: Tuple of `(name, value)` pairs for explicit keyword parameters.
        extra_args: Variadic positional arguments.
        extra_kwargs: Variadic keyword arguments.

    Returns:
        tuple[tuple[Tensor | TensorOperand, ...], dict[str, Tensor | TensorOperand]]:
            Combined `(args, kwargs)` pair.
    """
    packed_args = _pack_pos(pos_items, extra_args)
    packed_kw: dict[str, Tensor | TensorOperand] = dict(extra_kwargs)
    for k, v in kw_items:
        if v is not None:
            packed_kw[k] = v
    return packed_args, packed_kw


def _clean_kwargs(
    kwargs: dict[str, Tensor | TensorOperand],
) -> dict[str, int | float | bool | str | None | builtins.tuple[int, ...]]:
    """Convert keyword arguments into `TensorOrScalar`- and `ConfigValue`-compatible values.

    Args:
        kwargs: Input keyword dictionary.

    Returns:
        dict[str, int | float | bool | str | None | tuple[int, ...]]: Cleaned keyword dictionary.
    """
    out: dict[str, int | float | bool | str | None | builtins.tuple[int, ...]] = {}
    for k, v in kwargs.items():
        if k in _IGNORED_KW:
            continue
        key = "axis" if k == "dim" else k
        if isinstance(v, DType):
            out[key] = v.value
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out[key] = v
        elif isinstance(v, (builtins.tuple, builtins.list)):
            ints: list[int] = [x for x in v if isinstance(x, int)]
            if ints:
                out[key] = builtins.tuple(ints)
    return out


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
    "unstack",
    "where",
    "zeros",
    "zeros_like",
]


class math:
    """TensorFlow `math` namespace."""

    @staticmethod
    def abs(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.abs` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.abs(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def acos(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.acos` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.acos(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def acosh(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.acosh` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.acosh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def add(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.add` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.add(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_all(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_all` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.all(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_any(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_any` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.any(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def argmax(
        input: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        output_type: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.argmax` operation.

        Args:
            input: Operation parameter `input`.
            axis: Operation parameter `axis`.
            output_type: Operation parameter `output_type`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input,), (("axis", axis), ("output_type", output_type), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.argmax(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def argmin(
        input: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        output_type: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.argmin` operation.

        Args:
            input: Operation parameter `input`.
            axis: Operation parameter `axis`.
            output_type: Operation parameter `output_type`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input,), (("axis", axis), ("output_type", output_type), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.argmin(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def asin(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.asin` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.asin(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def asinh(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.asinh` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.asinh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def atan(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.atan` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.atan(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def atan2(
        y: Tensor | TensorOperand = None,
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.atan2` operation.

        Args:
            y: Operation parameter `y`.
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((y, x), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.atan2(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def atanh(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.atanh` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.atanh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def ceil(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.ceil` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.ceil(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def conj(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.conj` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.conj(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def cos(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.cos` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.cos(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def cosh(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.cosh` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.cosh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def count_nonzero(
        input: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        dtype: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.count_nonzero` operation.

        Args:
            input: Operation parameter `input`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            dtype: Operation parameter `dtype`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input,),
            (("axis", axis), ("keepdims", keepdims), ("dtype", dtype), ("name", name)),
            args,
            kwargs,
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.count_nonzero(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def digamma(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.digamma` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.digamma(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def divide(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.divide` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.divide(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def equal(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.equal` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.equal(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def erf(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.erf` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.erf(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def erfc(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.erfc` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.erfc(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def erfinv(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.erfinv` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.erfinv(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def exp(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.exp` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.exp(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def expm1(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.expm1` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.expm1(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def pow(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.pow` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.power(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def floor(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.floor` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.floor(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def greater(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.greater` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.greater(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def greater_equal(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.greater_equal` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.greater_equal(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def imag(
        input: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.imag` operation.

        Args:
            input: Operation parameter `input`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((input,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.imag(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def less(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.less` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.less(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def less_equal(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.less_equal` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.less_equal(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def lgamma(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.lgamma` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.lgamma(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def log(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.log` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.log(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def log1p(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.log1p` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.log1p(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def logical_and(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.logical_and` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.logical_and(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def logical_not(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.logical_not` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.logical_not(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def logical_or(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.logical_or` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.logical_or(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def logical_xor(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.logical_xor` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.logical_xor(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_max(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_max` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.max(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def maximum(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.maximum` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.maximum(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_mean(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_mean` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.mean(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_min(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_min` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.min(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def minimum(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.minimum` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.minimum(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def mod(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.mod` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.mod(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def multiply(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.multiply` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.multiply(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def negative(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.negative` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.negative(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def nextafter(
        x1: Tensor | TensorOperand = None,
        x2: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.nextafter` operation.

        Args:
            x1: Operation parameter `x1`.
            x2: Operation parameter `x2`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x1, x2), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.nextafter(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def not_equal(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.not_equal` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.not_equal(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_prod(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_prod` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.prod(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def real(
        input: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.real` operation.

        Args:
            input: Operation parameter `input`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((input,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.real(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reciprocal(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reciprocal` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.reciprocal(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def round(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.round` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.round(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def rsqrt(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.rsqrt` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.rsqrt(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def sign(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sign` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.sign(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def sin(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sin` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.sin(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def sinh(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sinh` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.sinh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def sqrt(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sqrt` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.sqrt(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def square(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.square` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.square(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_std(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_std` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.std(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def subtract(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.subtract` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.subtract(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_sum(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_sum` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.sum(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def tan(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.tan` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.tan(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def tanh(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.tanh` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.tanh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_variance(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_variance` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.variance(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def accumulate_n(
        inputs: Tensor | TensorOperand = None,
        shape: Tensor | TensorOperand = None,
        tensor_dtype: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.accumulate_n` operation.

        Args:
            inputs: Operation parameter `inputs`.
            shape: Operation parameter `shape`.
            tensor_dtype: Operation parameter `tensor_dtype`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (inputs,),
            (("shape", shape), ("tensor_dtype", tensor_dtype), ("name", name)),
            args,
            kwargs,
        )
        conv_args = [
            [_to_tensor(x) for x in a]
            if isinstance(a, (builtins.list, builtins.tuple))
            else _to_tensor(a)
            for a in args
        ]
        return _wrap_tensor(_ops.accumulate_n(*conv_args, **_clean_kwargs(kwargs)))

    @staticmethod
    def add_n(
        inputs: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.add_n` operation.

        Args:
            inputs: Operation parameter `inputs`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((inputs,), (("name", name),), args, kwargs)
        conv_args = [
            [_to_tensor(x) for x in a]
            if isinstance(a, (builtins.list, builtins.tuple))
            else _to_tensor(a)
            for a in args
        ]
        return _wrap_tensor(_ops.add_n(*conv_args, **_clean_kwargs(kwargs)))

    @staticmethod
    def angle(
        input: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.angle` operation.

        Args:
            input: Operation parameter `input`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((input,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.angle(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def approx_max_k(
        operand: Tensor | TensorOperand = None,
        k: Tensor | TensorOperand = None,
        reduction_dimension: Tensor | TensorOperand = None,
        recall_target: Tensor | TensorOperand = None,
        reduction_input_size_override: Tensor | TensorOperand = None,
        aggregate_to_topk: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.approx_max_k` operation.

        Args:
            operand: Operation parameter `operand`.
            k: Operation parameter `k`.
            reduction_dimension: Operation parameter `reduction_dimension`.
            recall_target: Operation parameter `recall_target`.
            reduction_input_size_override: Operation parameter `reduction_input_size_override`.
            aggregate_to_topk: Operation parameter `aggregate_to_topk`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (operand, k),
            (
                ("reduction_dimension", reduction_dimension),
                ("recall_target", recall_target),
                ("reduction_input_size_override", reduction_input_size_override),
                ("aggregate_to_topk", aggregate_to_topk),
                ("name", name),
            ),
            args,
            kwargs,
        )
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.approx_max_k(
                _to_tensor(args[0]),
                int(args[1]) if len(args) > 1 and isinstance(args[1], int) else 1,
            )
        )
        return _wrap_tensor(res)

    @staticmethod
    def approx_min_k(
        operand: Tensor | TensorOperand = None,
        k: Tensor | TensorOperand = None,
        reduction_dimension: Tensor | TensorOperand = None,
        recall_target: Tensor | TensorOperand = None,
        reduction_input_size_override: Tensor | TensorOperand = None,
        aggregate_to_topk: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.approx_min_k` operation.

        Args:
            operand: Operation parameter `operand`.
            k: Operation parameter `k`.
            reduction_dimension: Operation parameter `reduction_dimension`.
            recall_target: Operation parameter `recall_target`.
            reduction_input_size_override: Operation parameter `reduction_input_size_override`.
            aggregate_to_topk: Operation parameter `aggregate_to_topk`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (operand, k),
            (
                ("reduction_dimension", reduction_dimension),
                ("recall_target", recall_target),
                ("reduction_input_size_override", reduction_input_size_override),
                ("aggregate_to_topk", aggregate_to_topk),
                ("name", name),
            ),
            args,
            kwargs,
        )
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.approx_min_k(
                _to_tensor(args[0]),
                int(args[1]) if len(args) > 1 and isinstance(args[1], int) else 1,
            )
        )
        return _wrap_tensor(res)

    @staticmethod
    def bessel_i0(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.bessel_i0` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.bessel_i0(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def bessel_i0e(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.bessel_i0e` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.bessel_i0e(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def bessel_i1(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.bessel_i1` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.bessel_i1(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def bessel_i1e(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.bessel_i1e` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.bessel_i1e(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def betainc(
        a: Tensor | TensorOperand = None,
        b: Tensor | TensorOperand = None,
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.betainc` operation.

        Args:
            a: Operation parameter `a`.
            b: Operation parameter `b`.
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((a, b, x), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.betainc(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def bincount(
        arr: Tensor | TensorOperand = None,
        weights: Tensor | TensorOperand = None,
        minlength: Tensor | TensorOperand = None,
        maxlength: Tensor | TensorOperand = None,
        dtype: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        binary_output: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.bincount` operation.

        Args:
            arr: Operation parameter `arr`.
            weights: Operation parameter `weights`.
            minlength: Operation parameter `minlength`.
            maxlength: Operation parameter `maxlength`.
            dtype: Operation parameter `dtype`.
            name: Operation parameter `name`.
            axis: Operation parameter `axis`.
            binary_output: Operation parameter `binary_output`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (arr,),
            (
                ("weights", weights),
                ("minlength", minlength),
                ("maxlength", maxlength),
                ("dtype", dtype),
                ("name", name),
                ("axis", axis),
                ("binary_output", binary_output),
            ),
            args,
            kwargs,
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _k_ops.bincount(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def confusion_matrix(
        labels: Tensor | TensorOperand = None,
        predictions: Tensor | TensorOperand = None,
        num_classes: Tensor | TensorOperand = None,
        weights: Tensor | TensorOperand = None,
        dtype: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.confusion_matrix` operation.

        Args:
            labels: Operation parameter `labels`.
            predictions: Operation parameter `predictions`.
            num_classes: Operation parameter `num_classes`.
            weights: Operation parameter `weights`.
            dtype: Operation parameter `dtype`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (labels, predictions, num_classes),
            (("weights", weights), ("dtype", dtype), ("name", name)),
            args,
            kwargs,
        )
        return _wrap_tensor(
            as_operand(
                _ops.confusion_matrix(
                    _to_tensor(args[0]),
                    _to_tensor(args[1] if len(args) > 1 else args[0]),
                    int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 2,
                )
            )
        )

    @staticmethod
    def cumprod(
        x: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        exclusive: Tensor | TensorOperand = None,
        reverse: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.cumprod` operation.

        Args:
            x: Operation parameter `x`.
            axis: Operation parameter `axis`.
            exclusive: Operation parameter `exclusive`.
            reverse: Operation parameter `reverse`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (x,),
            (("axis", axis), ("exclusive", exclusive), ("reverse", reverse), ("name", name)),
            args,
            kwargs,
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.cumprod(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def cumsum(
        x: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        exclusive: Tensor | TensorOperand = None,
        reverse: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.cumsum` operation.

        Args:
            x: Operation parameter `x`.
            axis: Operation parameter `axis`.
            exclusive: Operation parameter `exclusive`.
            reverse: Operation parameter `reverse`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (x,),
            (("axis", axis), ("exclusive", exclusive), ("reverse", reverse), ("name", name)),
            args,
            kwargs,
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.cumsum(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def cumulative_logsumexp(
        x: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        exclusive: Tensor | TensorOperand = None,
        reverse: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.cumulative_logsumexp` operation.

        Args:
            x: Operation parameter `x`.
            axis: Operation parameter `axis`.
            exclusive: Operation parameter `exclusive`.
            reverse: Operation parameter `reverse`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (x,),
            (("axis", axis), ("exclusive", exclusive), ("reverse", reverse), ("name", name)),
            args,
            kwargs,
        )
        return _wrap_tensor(
            _ops.cumulative_logsumexp(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def divide_no_nan(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.divide_no_nan` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        return _wrap_tensor(
            as_operand(_ops.divide_no_nan(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs)))
        )

    @staticmethod
    def erfcinv(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.erfcinv` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.erfcinv(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def floordiv(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.floordiv` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.floor_divide(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def floormod(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.floormod` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.mod(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def igamma(
        a: Tensor | TensorOperand = None,
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.igamma` operation.

        Args:
            a: Operation parameter `a`.
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((a, x), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.igamma(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def igammac(
        a: Tensor | TensorOperand = None,
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.igammac` operation.

        Args:
            a: Operation parameter `a`.
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((a, x), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.igammac(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def in_top_k(
        targets: Tensor | TensorOperand = None,
        predictions: Tensor | TensorOperand = None,
        k: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.in_top_k` operation.

        Args:
            targets: Operation parameter `targets`.
            predictions: Operation parameter `predictions`.
            k: Operation parameter `k`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((targets, predictions, k), (("name", name),), args, kwargs)
        return _wrap_tensor(
            _k_ops.in_top_k(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def invert_permutation(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.invert_permutation` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _k_ops.argsort(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def is_finite(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.is_finite` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.isfinite(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def is_inf(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.is_inf` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.isinf(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def is_nan(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.is_nan` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.isnan(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def is_non_decreasing(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.is_non_decreasing` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        return _wrap_tensor(
            _ops.is_non_decreasing(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def is_strictly_increasing(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.is_strictly_increasing` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        return _wrap_tensor(
            _ops.is_strictly_increasing(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def l2_normalize(
        x: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        epsilon: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        dim: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.l2_normalize` operation.

        Args:
            x: Operation parameter `x`.
            axis: Operation parameter `axis`.
            epsilon: Operation parameter `epsilon`.
            name: Operation parameter `name`.
            dim: Operation parameter `dim`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (x,), (("axis", axis), ("epsilon", epsilon), ("name", name), ("dim", dim)), args, kwargs
        )
        return _wrap_tensor(args[0] if args else 0.0)

    @staticmethod
    def lbeta(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.lbeta` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.lbeta(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def log_sigmoid(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.log_sigmoid` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        return _wrap_tensor(
            _k_ops.log_sigmoid(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def log_softmax(
        logits: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.log_softmax` operation.

        Args:
            logits: Operation parameter `logits`.
            axis: Operation parameter `axis`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((logits,), (("axis", axis), ("name", name)), args, kwargs)
        return _wrap_tensor(
            _k_ops.log_softmax(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def multiply_no_nan(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.multiply_no_nan` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        return _wrap_tensor(
            _ops.multiply_no_nan(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def ndtri(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.ndtri` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.ndtri(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def polygamma(
        a: Tensor | TensorOperand = None,
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.polygamma` operation.

        Args:
            a: Operation parameter `a`.
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((a, x), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.polygamma(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def polyval(
        coeffs: Tensor | TensorOperand = None,
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.polyval` operation.

        Args:
            coeffs: Operation parameter `coeffs`.
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((coeffs, x), (("name", name),), args, kwargs)
        _clean_kwargs(kwargs)
        res = args[0] if args else 0.0
        return _wrap_tensor(res)

    @staticmethod
    def reciprocal_no_nan(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reciprocal_no_nan` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        return _wrap_tensor(
            _ops.reciprocal_no_nan(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def reduce_euclidean_norm(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_euclidean_norm` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        return _wrap_tensor(
            _ops.reduce_euclidean_norm(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def reduce_logsumexp(
        input_tensor: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        keepdims: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_logsumexp` operation.

        Args:
            input_tensor: Operation parameter `input_tensor`.
            axis: Operation parameter `axis`.
            keepdims: Operation parameter `keepdims`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input_tensor,), (("axis", axis), ("keepdims", keepdims), ("name", name)), args, kwargs
        )
        return _wrap_tensor(
            _ops.reduce_logsumexp(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def rint(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.rint` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.rint(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def scalar_mul(
        scalar: Tensor | TensorOperand = None,
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.scalar_mul` operation.

        Args:
            scalar: Operation parameter `scalar`.
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((scalar, x), (("name", name),), args, kwargs)
        return _wrap_tensor(_ops.multiply(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs)))

    @staticmethod
    def segment_max(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.segment_max` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((data, segment_ids), (("name", name),), args, kwargs)
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.segment_max(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def segment_mean(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.segment_mean` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((data, segment_ids), (("name", name),), args, kwargs)
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.segment_mean(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def segment_min(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.segment_min` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((data, segment_ids), (("name", name),), args, kwargs)
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.segment_min(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def segment_prod(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.segment_prod` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((data, segment_ids), (("name", name),), args, kwargs)
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.segment_prod(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def segment_sum(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.segment_sum` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((data, segment_ids), (("name", name),), args, kwargs)
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.segment_sum(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def sigmoid(
        x: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sigmoid` operation.

        Args:
            x: Operation parameter `x`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x,), (("name", name),), args, kwargs)
        from . import nn

        x = args[0] if args else None
        return nn.sigmoid(x)

    @staticmethod
    def sobol_sample(
        dim: Tensor | TensorOperand = None,
        num_results: Tensor | TensorOperand = None,
        skip: Tensor | TensorOperand = None,
        dtype: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sobol_sample` operation.

        Args:
            dim: Operation parameter `dim`.
            num_results: Operation parameter `num_results`.
            skip: Operation parameter `skip`.
            dtype: Operation parameter `dtype`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (dim, num_results), (("skip", skip), ("dtype", dtype), ("name", name)), args, kwargs
        )
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.sobol_sample(
                int(args[0]) if args and isinstance(args[0], int) else 1,
                int(args[1]) if len(args) > 1 and isinstance(args[1], int) else 1,
            )
        )
        return _wrap_tensor(res)

    @staticmethod
    def softmax(
        logits: Tensor | TensorOperand = None,
        axis: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.softmax` operation.

        Args:
            logits: Operation parameter `logits`.
            axis: Operation parameter `axis`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((logits,), (("axis", axis), ("name", name)), args, kwargs)
        from . import nn

        logits = args[0] if args else None
        axis_raw = args[1] if len(args) > 1 else kwargs.get("axis")
        axis = int(axis_raw) if isinstance(axis_raw, int) else None
        return nn.softmax(logits, axis=axis)

    @staticmethod
    def softplus(
        features: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.softplus` operation.

        Args:
            features: Operation parameter `features`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((features,), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _k_ops.softplus(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def softsign(
        features: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.softsign` operation.

        Args:
            features: Operation parameter `features`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((features,), (("name", name),), args, kwargs)
        return _wrap_tensor(
            _k_ops.softsign(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def squared_difference(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.squared_difference` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        return _wrap_tensor(
            _ops.squared_difference(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def top_k(
        input: Tensor | TensorOperand = None,
        k: Tensor | TensorOperand = None,
        sorted: Tensor | TensorOperand = None,
        index_type: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor] | TensorOperand:
        """Execute `math.top_k` operation.

        Args:
            input: Operation parameter `input`.
            k: Operation parameter `k`.
            sorted: Operation parameter `sorted`.
            index_type: Operation parameter `index_type`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (input, k),
            (("sorted", sorted), ("index_type", index_type), ("name", name)),
            args,
            kwargs,
        )
        clean_kw = _clean_kwargs(kwargs)
        res = _k_ops.top_k(_to_tensor(args[0]), *args[1:], **clean_kw)
        return _wrap(res)

    @staticmethod
    def truediv(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.truediv` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.true_divide(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def unsorted_segment_max(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        num_segments: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_max` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            num_segments: Operation parameter `num_segments`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (data, segment_ids, num_segments), (("name", name),), args, kwargs
        )
        del kwargs
        num_seg = int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 1
        res = as_operand(
            _ops.unsorted_segment_max(
                _to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]), num_seg
            )
        )
        return _wrap_tensor(res)

    @staticmethod
    def unsorted_segment_mean(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        num_segments: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_mean` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            num_segments: Operation parameter `num_segments`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (data, segment_ids, num_segments), (("name", name),), args, kwargs
        )
        del kwargs
        num_seg = int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 1
        res = as_operand(
            _ops.unsorted_segment_mean(
                _to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]), num_seg
            )
        )
        return _wrap_tensor(res)

    @staticmethod
    def unsorted_segment_min(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        num_segments: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_min` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            num_segments: Operation parameter `num_segments`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (data, segment_ids, num_segments), (("name", name),), args, kwargs
        )
        del kwargs
        num_seg = int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 1
        res = as_operand(
            _ops.unsorted_segment_min(
                _to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]), num_seg
            )
        )
        return _wrap_tensor(res)

    @staticmethod
    def unsorted_segment_prod(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        num_segments: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_prod` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            num_segments: Operation parameter `num_segments`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (data, segment_ids, num_segments), (("name", name),), args, kwargs
        )
        del kwargs
        num_seg = int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 1
        res = as_operand(
            _ops.unsorted_segment_prod(
                _to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]), num_seg
            )
        )
        return _wrap_tensor(res)

    @staticmethod
    def unsorted_segment_sqrt_n(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        num_segments: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_sqrt_n` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            num_segments: Operation parameter `num_segments`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (data, segment_ids, num_segments), (("name", name),), args, kwargs
        )
        del kwargs
        num_seg = int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 1
        res = as_operand(
            _ops.unsorted_segment_sqrt_n(
                _to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]), num_seg
            )
        )
        return _wrap_tensor(res)

    @staticmethod
    def unsorted_segment_sum(
        data: Tensor | TensorOperand = None,
        segment_ids: Tensor | TensorOperand = None,
        num_segments: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_sum` operation.

        Args:
            data: Operation parameter `data`.
            segment_ids: Operation parameter `segment_ids`.
            num_segments: Operation parameter `num_segments`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call(
            (data, segment_ids, num_segments), (("name", name),), args, kwargs
        )
        del kwargs
        num_seg = int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 1
        res = as_operand(
            _ops.unsorted_segment_sum(
                _to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]), num_seg
            )
        )
        return _wrap_tensor(res)

    @staticmethod
    def xdivy(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.xdivy` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        return _wrap_tensor(_ops.xdivy(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs)))

    @staticmethod
    def xlog1py(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.xlog1py` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        return _wrap_tensor(_ops.xlog1py(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs)))

    @staticmethod
    def xlogy(
        x: Tensor | TensorOperand = None,
        y: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.xlogy` operation.

        Args:
            x: Operation parameter `x`.
            y: Operation parameter `y`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, y), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.xlogy(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def zero_fraction(
        value: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.zero_fraction` operation.

        Args:
            value: Operation parameter `value`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((value,), (("name", name),), args, kwargs)
        return _wrap_tensor(
            _ops.zero_fraction(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def zeta(
        x: Tensor | TensorOperand = None,
        q: Tensor | TensorOperand = None,
        name: Tensor | TensorOperand = None,
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.zeta` operation.

        Args:
            x: Operation parameter `x`.
            q: Operation parameter `q`.
            name: Operation parameter `name`.
            *args: Additional positional operation arguments.
            **kwargs: Additional keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        (args, kwargs) = _pack_call((x, q), (("name", name),), args, kwargs)
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.zeta(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)


abs = math.abs
acos = math.acos
acosh = math.acosh
add = math.add
reduce_all = math.reduce_all
reduce_any = math.reduce_any


def range(
    start: Tensor | TensorOperand = None,
    limit: Tensor | TensorOperand = None,
    delta: Tensor | TensorOperand = None,
    dtype: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `range` operation.

    Args:
        start: Operation parameter `start`.
        limit: Operation parameter `limit`.
        delta: Operation parameter `delta`.
        dtype: Operation parameter `dtype`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (start, limit, delta), (("dtype", dtype), ("name", name)), args, kwargs
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.arange(*args, **clean_kw)
    return _wrap_tensor(res)


argmax = math.argmax
argmin = math.argmin
asin = math.asin
asinh = math.asinh
atan = math.atan
atan2 = math.atan2
atanh = math.atanh


def bitcast(
    input: Tensor | TensorOperand = None,
    type: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `bitcast` operation.

    Args:
        input: Operation parameter `input`.
        type: Operation parameter `type`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((input,), (("type", type), ("name", name)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.bitcast(_to_tensor(args[0]), **clean_kw)
    return _wrap_tensor(res)


def broadcast_to(
    input: Tensor | TensorOperand = None,
    shape: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `broadcast_to` operation.

    Args:
        input: Operation parameter `input`.
        shape: Operation parameter `shape`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((input, shape), (("name", name),), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.broadcast_to(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def cast(
    x: Tensor | TensorOperand = None,
    dtype: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `cast` operation.

    Args:
        x: Operation parameter `x`.
        dtype: Operation parameter `dtype`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((x, dtype), (("name", name),), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.cast(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def concat(
    values: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `concat` operation.

    Args:
        values: Operation parameter `values`.
        axis: Operation parameter `axis`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((values,), (("axis", axis), ("name", name)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    first = (
        [_to_tensor(item) for item in args[0]]
        if args and isinstance(args[0], (builtins.list, builtins.tuple))
        else [_to_tensor(args[0])]
    )
    res = _k_ops.concatenate(first, *args[1:], **clean_kw)
    return _wrap_tensor(res)


cos = math.cos
cosh = math.cosh
divide = math.divide


def einsum(
    equation: Tensor | TensorOperand = None,
    *inputs: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `einsum` operation.

    Args:
        equation: Contraction equation string.
        *inputs: Input tensor operands.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Operation result.
    """
    args = _pack_pos((equation,), inputs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.einsum(*[_to_tensor(a) if not isinstance(a, str) else a for a in args], **clean_kw)
    return _wrap_tensor(res)


equal = math.equal
exp = math.exp


def expand_dims(
    input: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `expand_dims` operation.

    Args:
        input: Operation parameter `input`.
        axis: Operation parameter `axis`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((input, axis), (("name", name),), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.expand_dims(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def eye(
    num_rows: Tensor | TensorOperand = None,
    num_columns: Tensor | TensorOperand = None,
    batch_shape: Tensor | TensorOperand = None,
    dtype: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `eye` operation.

    Args:
        num_rows: Operation parameter `num_rows`.
        num_columns: Operation parameter `num_columns`.
        batch_shape: Operation parameter `batch_shape`.
        dtype: Operation parameter `dtype`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (num_rows, num_columns),
        (("batch_shape", batch_shape), ("dtype", dtype), ("name", name)),
        args,
        kwargs,
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.eye(*args, **clean_kw)
    return _wrap_tensor(res)


def reshape(
    tensor: Tensor | TensorOperand = None,
    shape: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `reshape` operation.

    Args:
        tensor: Operation parameter `tensor`.
        shape: Operation parameter `shape`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((tensor, shape), (("name", name),), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.reshape(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


pow = math.pow
floor = math.floor


def fill(
    dims: Tensor | TensorOperand = None,
    value: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    layout: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `fill` operation.

    Args:
        dims: Operation parameter `dims`.
        value: Operation parameter `value`.
        name: Operation parameter `name`.
        layout: Operation parameter `layout`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((dims, value), (("name", name), ("layout", layout)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.full(*args, **clean_kw)
    return _wrap_tensor(res)


def gather(
    params: Tensor | TensorOperand = None,
    indices: Tensor | TensorOperand = None,
    validate_indices: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    batch_dims: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `gather` operation.

    Args:
        params: Operation parameter `params`.
        indices: Operation parameter `indices`.
        validate_indices: Operation parameter `validate_indices`.
        axis: Operation parameter `axis`.
        batch_dims: Operation parameter `batch_dims`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (params, indices),
        (
            ("validate_indices", validate_indices),
            ("axis", axis),
            ("batch_dims", batch_dims),
            ("name", name),
        ),
        args,
        kwargs,
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.take(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def gather_nd(
    params: Tensor | TensorOperand = None,
    indices: Tensor | TensorOperand = None,
    batch_dims: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    bad_indices_policy: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `gather_nd` operation.

    Args:
        params: Operation parameter `params`.
        indices: Operation parameter `indices`.
        batch_dims: Operation parameter `batch_dims`.
        name: Operation parameter `name`.
        bad_indices_policy: Operation parameter `bad_indices_policy`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (params, indices),
        (("batch_dims", batch_dims), ("name", name), ("bad_indices_policy", bad_indices_policy)),
        args,
        kwargs,
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.take(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


greater = math.greater
greater_equal = math.greater_equal


def identity(
    input: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `identity` operation.

    Args:
        input: Operation parameter `input`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((input,), (("name", name),), args, kwargs)
    del kwargs
    res = _to_tensor(args[0]) if args else _to_tensor(0.0)
    return _wrap_tensor(res)


less = math.less
less_equal = math.less_equal


def linspace(
    start: Tensor | TensorOperand = None,
    stop: Tensor | TensorOperand = None,
    num: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `linspace` operation.

    Args:
        start: Operation parameter `start`.
        stop: Operation parameter `stop`.
        num: Operation parameter `num`.
        name: Operation parameter `name`.
        axis: Operation parameter `axis`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((start, stop, num), (("name", name), ("axis", axis)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.linspace(*args, **clean_kw)
    return _wrap_tensor(res)


logical_and = math.logical_and
logical_not = math.logical_not
logical_or = math.logical_or


def matmul(
    a: Tensor | TensorOperand = None,
    b: Tensor | TensorOperand = None,
    transpose_a: Tensor | TensorOperand = None,
    transpose_b: Tensor | TensorOperand = None,
    adjoint_a: Tensor | TensorOperand = None,
    adjoint_b: Tensor | TensorOperand = None,
    a_is_sparse: Tensor | TensorOperand = None,
    b_is_sparse: Tensor | TensorOperand = None,
    output_type: Tensor | TensorOperand = None,
    grad_a: Tensor | TensorOperand = None,
    grad_b: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `matmul` operation.

    Args:
        a: Operation parameter `a`.
        b: Operation parameter `b`.
        transpose_a: Operation parameter `transpose_a`.
        transpose_b: Operation parameter `transpose_b`.
        adjoint_a: Operation parameter `adjoint_a`.
        adjoint_b: Operation parameter `adjoint_b`.
        a_is_sparse: Operation parameter `a_is_sparse`.
        b_is_sparse: Operation parameter `b_is_sparse`.
        output_type: Operation parameter `output_type`.
        grad_a: Operation parameter `grad_a`.
        grad_b: Operation parameter `grad_b`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (a, b),
        (
            ("transpose_a", transpose_a),
            ("transpose_b", transpose_b),
            ("adjoint_a", adjoint_a),
            ("adjoint_b", adjoint_b),
            ("a_is_sparse", a_is_sparse),
            ("b_is_sparse", b_is_sparse),
            ("output_type", output_type),
            ("grad_a", grad_a),
            ("grad_b", grad_b),
            ("name", name),
        ),
        args,
        kwargs,
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.matmul(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


reduce_max = math.reduce_max
maximum = math.maximum
reduce_mean = math.reduce_mean


def meshgrid(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
    """Execute `meshgrid` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        tuple[Tensor, ...] | list[Tensor]: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.meshgrid(*[_to_tensor(a) for a in args], **clean_kw)
    wrapped = _wrap(res)
    return wrapped if isinstance(wrapped, (builtins.tuple, builtins.list)) else [wrapped]


reduce_min = math.reduce_min
minimum = math.minimum
multiply = math.multiply
negative = math.negative


def norm(
    tensor: Tensor | TensorOperand = None,
    ord: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    keepdims: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `norm` operation.

    Args:
        tensor: Operation parameter `tensor`.
        ord: Operation parameter `ord`.
        axis: Operation parameter `axis`.
        keepdims: Operation parameter `keepdims`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (tensor,),
        (("ord", ord), ("axis", axis), ("keepdims", keepdims), ("name", name)),
        args,
        kwargs,
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.norm(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


not_equal = math.not_equal


def ones(
    shape: Tensor | TensorOperand = None,
    dtype: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    layout: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `ones` operation.

    Args:
        shape: Operation parameter `shape`.
        dtype: Operation parameter `dtype`.
        name: Operation parameter `name`.
        layout: Operation parameter `layout`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (shape,), (("dtype", dtype), ("name", name), ("layout", layout)), args, kwargs
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.ones(*args, **clean_kw)
    return _wrap_tensor(res)


def ones_like(
    input: Tensor | TensorOperand = None,
    dtype: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    layout: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `ones_like` operation.

    Args:
        input: Operation parameter `input`.
        dtype: Operation parameter `dtype`.
        name: Operation parameter `name`.
        layout: Operation parameter `layout`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (input,), (("dtype", dtype), ("name", name), ("layout", layout)), args, kwargs
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.ones_like(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


reduce_prod = math.reduce_prod


def repeat(
    input: Tensor | TensorOperand = None,
    repeats: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `repeat` operation.

    Args:
        input: Operation parameter `input`.
        repeats: Operation parameter `repeats`.
        axis: Operation parameter `axis`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((input, repeats), (("axis", axis), ("name", name)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.repeat(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def roll(
    input: Tensor | TensorOperand = None,
    shift: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `roll` operation.

    Args:
        input: Operation parameter `input`.
        shift: Operation parameter `shift`.
        axis: Operation parameter `axis`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((input, shift), (("axis", axis), ("name", name)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.roll(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


round = math.round


def scatter_nd(
    indices: Tensor | TensorOperand = None,
    updates: Tensor | TensorOperand = None,
    shape: Tensor | TensorOperand = None,
    bad_indices_policy: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `scatter_nd` operation.

    Args:
        indices: Operation parameter `indices`.
        updates: Operation parameter `updates`.
        shape: Operation parameter `shape`.
        bad_indices_policy: Operation parameter `bad_indices_policy`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (indices, updates, shape),
        (("bad_indices_policy", bad_indices_policy), ("name", name)),
        args,
        kwargs,
    )
    del kwargs
    updates_t = _to_tensor(args[1] if len(args) > 1 else 0.0)
    shape_arg = args[2] if len(args) > 2 else updates_t.shape
    base = _k_ops.zeros(shape_arg, dtype=updates_t.dtype)
    idx_t = _to_tensor(args[0] if args else [0])
    res = _k_ops.scatter(base, 0, idx_t, updates_t)
    return _wrap_tensor(res)


def shape(
    input: Tensor | TensorOperand = None,
    out_type: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> builtins.tuple[Tensor, ...]:
    """Execute `shape` operation.

    Args:
        input: Operation parameter `input`.
        out_type: Operation parameter `out_type`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((input,), (("out_type", out_type), ("name", name)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.shape(*[_to_tensor(a) for a in args], **clean_kw)
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
    input_: Tensor | TensorOperand = None,
    begin: Tensor | TensorOperand = None,
    size: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `slice` operation.

    Args:
        input_: Operation parameter `input_`.
        begin: Operation parameter `begin`.
        size: Operation parameter `size`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((input_, begin, size), (("name", name),), args, kwargs)
    if (
        len(args) == 3
        and isinstance(args[1], (builtins.list, builtins.tuple))
        and isinstance(args[2], (builtins.list, builtins.tuple))
    ):
        curr = _to_tensor(args[0])
        for axis, (b, sz) in enumerate(zip(args[1], args[2])):
            b_int = int(b) if isinstance(b, int) else 0
            sz_int = int(sz) if isinstance(sz, int) else 0
            end_int = None if sz_int < 0 else b_int + sz_int
            curr = _to_tensor(_k_ops.slice(curr, axis, b_int, end_int, 1))
        return _wrap_tensor(curr)
    res = _k_ops.slice(
        *[_to_tensor(a) if i == 0 else a for (i, a) in enumerate(args)], **_clean_kwargs(kwargs)
    )
    return _wrap_tensor(res)


def split(
    value: Tensor | TensorOperand = None,
    num_or_size_splits: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    num: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> builtins.list[Tensor]:
    """Execute `split` operation.

    Args:
        value: Operation parameter `value`.
        num_or_size_splits: Operation parameter `num_or_size_splits`.
        axis: Operation parameter `axis`.
        num: Operation parameter `num`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (value, num_or_size_splits), (("axis", axis), ("num", num), ("name", name)), args, kwargs
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.split(_to_tensor(args[0]), *args[1:], **clean_kw)
    wrapped = _wrap(res)
    return (
        builtins.list(wrapped)
        if isinstance(wrapped, (builtins.list, builtins.tuple))
        else [wrapped]
    )


sqrt = math.sqrt
square = math.square


def squeeze(
    input: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `squeeze` operation.

    Args:
        input: Operation parameter `input`.
        axis: Operation parameter `axis`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((input,), (("axis", axis), ("name", name)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.squeeze(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def stack(
    values: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `stack` operation.

    Args:
        values: Operation parameter `values`.
        axis: Operation parameter `axis`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((values,), (("axis", axis), ("name", name)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    first = (
        [_to_tensor(item) for item in args[0]]
        if args and isinstance(args[0], (builtins.list, builtins.tuple))
        else [_to_tensor(args[0])]
    )
    res = _k_ops.stack(first, *args[1:], **clean_kw)
    return _wrap_tensor(res)


def strided_slice(
    input_: Tensor | TensorOperand = None,
    begin: Tensor | TensorOperand = None,
    end: Tensor | TensorOperand = None,
    strides: Tensor | TensorOperand = None,
    begin_mask: Tensor | TensorOperand = None,
    end_mask: Tensor | TensorOperand = None,
    ellipsis_mask: Tensor | TensorOperand = None,
    new_axis_mask: Tensor | TensorOperand = None,
    shrink_axis_mask: Tensor | TensorOperand = None,
    var: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `strided_slice` operation.

    Args:
        input_: Operation parameter `input_`.
        begin: Operation parameter `begin`.
        end: Operation parameter `end`.
        strides: Operation parameter `strides`.
        begin_mask: Operation parameter `begin_mask`.
        end_mask: Operation parameter `end_mask`.
        ellipsis_mask: Operation parameter `ellipsis_mask`.
        new_axis_mask: Operation parameter `new_axis_mask`.
        shrink_axis_mask: Operation parameter `shrink_axis_mask`.
        var: Operation parameter `var`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (input_, begin, end, strides),
        (
            ("begin_mask", begin_mask),
            ("end_mask", end_mask),
            ("ellipsis_mask", ellipsis_mask),
            ("new_axis_mask", new_axis_mask),
            ("shrink_axis_mask", shrink_axis_mask),
            ("var", var),
            ("name", name),
        ),
        args,
        kwargs,
    )
    del kwargs
    if (
        len(args) >= 3
        and isinstance(args[1], (builtins.list, builtins.tuple))
        and isinstance(args[2], (builtins.list, builtins.tuple))
    ):
        curr = _to_tensor(args[0])
        strides_seq = (
            args[3]
            if len(args) >= 4 and isinstance(args[3], (builtins.list, builtins.tuple))
            else [1] * len(args[1])
        )
        for axis, (b, e, st) in enumerate(zip(args[1], args[2], strides_seq)):
            b_int = int(b) if isinstance(b, int) else 0
            e_int = int(e) if isinstance(e, int) else 0
            st_int = int(st) if isinstance(st, int) else 1
            curr = _to_tensor(_k_ops.slice(curr, axis, b_int, e_int, st_int))
        return _wrap_tensor(curr)
    return _wrap_tensor(args[0] if args else 0.0)


subtract = math.subtract
reduce_sum = math.reduce_sum
tan = math.tan
tanh = math.tanh


def tile(
    input: Tensor | TensorOperand = None,
    multiples: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `tile` operation.

    Args:
        input: Operation parameter `input`.
        multiples: Operation parameter `multiples`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((input, multiples), (("name", name),), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.tile(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def transpose(
    a: Tensor | TensorOperand = None,
    perm: Tensor | TensorOperand = None,
    conjugate: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `transpose` operation.

    Args:
        a: Operation parameter `a`.
        perm: Operation parameter `perm`.
        conjugate: Operation parameter `conjugate`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((a, perm), (("conjugate", conjugate), ("name", name)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.transpose(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def unstack(
    value: Tensor | TensorOperand = None,
    num: Tensor | TensorOperand = None,
    axis: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> builtins.list[Tensor]:
    """Execute `unstack` operation.

    Args:
        value: Operation parameter `value`.
        num: Operation parameter `num`.
        axis: Operation parameter `axis`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (value,), (("num", num), ("axis", axis), ("name", name)), args, kwargs
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.unstack(_to_tensor(args[0]), *args[1:], **clean_kw)
    wrapped = _wrap(res)
    return (
        builtins.list(wrapped)
        if isinstance(wrapped, (builtins.list, builtins.tuple))
        else [wrapped]
    )


def where(
    condition: Tensor | TensorOperand = None,
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `where` operation.

    Args:
        condition: Operation parameter `condition`.
        x: Operation parameter `x`.
        y: Operation parameter `y`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((condition, x, y), (("name", name),), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.where(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def zeros(
    shape: Tensor | TensorOperand = None,
    dtype: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    layout: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `zeros` operation.

    Args:
        shape: Operation parameter `shape`.
        dtype: Operation parameter `dtype`.
        name: Operation parameter `name`.
        layout: Operation parameter `layout`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (shape,), (("dtype", dtype), ("name", name), ("layout", layout)), args, kwargs
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.zeros(*args, **clean_kw)
    return _wrap_tensor(res)


def zeros_like(
    input: Tensor | TensorOperand = None,
    dtype: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    layout: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `zeros_like` operation.

    Args:
        input: Operation parameter `input`.
        dtype: Operation parameter `dtype`.
        name: Operation parameter `name`.
        layout: Operation parameter `layout`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call(
        (input,), (("dtype", dtype), ("name", name), ("layout", layout)), args, kwargs
    )
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.zeros_like(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def tensordot(
    a: Tensor | TensorOperand = None,
    b: Tensor | TensorOperand = None,
    axes: Tensor | TensorOperand = None,
    name: Tensor | TensorOperand = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `tensordot` operation.

    Args:
        a: Operation parameter `a`.
        b: Operation parameter `b`.
        axes: Operation parameter `axes`.
        name: Operation parameter `name`.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    (args, kwargs) = _pack_call((a, b), (("axes", axes), ("name", name)), args, kwargs)
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.tensordot(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


scalar_mul = math.scalar_mul
truediv = math.truediv
