"""TensorFlow `math` namespace and top-level mathematical/array operations."""

from __future__ import annotations

import builtins

from zero_tensorflow._typing import TensorOperand, as_operand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap, _wrap_tensor

import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.core.shape import TensorOrScalar

from zero_keras import ops as _k_ops  # isort: skip


def _clean_kwargs(kwargs: dict[str, Tensor | TensorOperand]) -> dict[str, TensorOrScalar]:
    """Convert keyword arguments into `TensorOrScalar`-compatible values.

    Args:
        kwargs: Input keyword dictionary.

    Returns:
        dict[str, TensorOrScalar]: Cleaned keyword dictionary.
    """
    return {
        ("axis" if k == "dim" else k): v
        for k, v in kwargs.items()
        if isinstance(v, (int, float, bool, str))
    }


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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.abs` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.abs(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def acos(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.acos` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.acos(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def acosh(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.acosh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.acosh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def add(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.add` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.add(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_all(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_all` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.all(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_any(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_any` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.any(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def argmax(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.argmax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.argmax(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def argmin(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.argmin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.argmin(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def asin(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.asin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.asin(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def asinh(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.asinh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.asinh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def atan(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.atan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.atan(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def atan2(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.atan2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.atan2(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def atanh(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.atanh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.atanh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def ceil(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.ceil` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.ceil(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def conj(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.conj` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.conj(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def cos(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.cos` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.cos(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def cosh(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.cosh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.cosh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def count_nonzero(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.count_nonzero` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.count_nonzero(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def digamma(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.digamma` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.digamma(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def divide(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.divide` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.divide(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def equal(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.equal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.equal(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def erf(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.erf` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.erf(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def erfc(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.erfc` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.erfc(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def erfinv(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.erfinv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.erfinv(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def exp(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.exp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.exp(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def expm1(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.expm1` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.expm1(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def pow(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.pow` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.power(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def floor(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.floor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.floor(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def greater(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.greater` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.greater(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def greater_equal(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.greater_equal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.greater_equal(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def imag(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.imag` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.imag(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def less(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.less` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.less(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def less_equal(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.less_equal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.less_equal(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def lgamma(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.lgamma` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.lgamma(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def log(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.log` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.log(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def log1p(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.log1p` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.log1p(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def logical_and(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.logical_and` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.logical_and(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def logical_not(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.logical_not` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.logical_not(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def logical_or(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.logical_or` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.logical_or(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def logical_xor(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.logical_xor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.logical_xor(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_max(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_max` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.max(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def maximum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.maximum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.maximum(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_mean(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_mean` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.mean(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_min(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_min` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.min(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def minimum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.minimum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.minimum(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def mod(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.mod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.mod(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def multiply(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.multiply` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.multiply(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def negative(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.negative` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.negative(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def nextafter(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.nextafter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.nextafter(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def not_equal(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.not_equal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.not_equal(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_prod(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_prod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.prod(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def real(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.real` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.real(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reciprocal(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reciprocal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.reciprocal(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def round(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.round` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.round(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def rsqrt(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.rsqrt` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.rsqrt(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def sign(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.sign(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def sin(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.sin(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def sinh(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sinh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.sinh(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def sqrt(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sqrt` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.sqrt(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def square(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.square` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.square(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_std(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_std` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.std(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def subtract(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.subtract` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.subtract(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_sum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_sum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.sum(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def tan(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.tan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.tan(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def tanh(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.tanh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _k_ops.tensordot(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def reduce_variance(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_variance` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.variance(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def accumulate_n(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.accumulate_n` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.accumulate_n(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def add_n(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.add_n` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(_ops.add_n(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs)))

    @staticmethod
    def angle(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.angle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.angle(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def approx_max_k(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.approx_max_k` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.approx_min_k` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.bessel_i0` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.bessel_i0(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def bessel_i0e(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.bessel_i0e` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.bessel_i0e(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def bessel_i1(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.bessel_i1` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.bessel_i1(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def bessel_i1e(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.bessel_i1e` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.bessel_i1e(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def betainc(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.betainc` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.betainc(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def bincount(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.bincount` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _k_ops.bincount(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def confusion_matrix(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.confusion_matrix` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.cumprod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.cumprod(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def cumsum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.cumsum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.cumsum(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def cumulative_logsumexp(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.cumulative_logsumexp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.cumulative_logsumexp(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def divide_no_nan(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.divide_no_nan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            as_operand(_ops.divide_no_nan(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs)))
        )

    @staticmethod
    def erfcinv(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.erfcinv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.erfcinv(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def floordiv(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.floordiv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.floor_divide(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def floormod(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.floormod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.mod(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def igamma(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.igamma` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.igamma(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def igammac(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.igammac` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.igammac(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def in_top_k(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.in_top_k` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _k_ops.in_top_k(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def invert_permutation(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.invert_permutation` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _k_ops.argsort(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def is_finite(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.is_finite` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.isfinite(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def is_inf(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.is_inf` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.isinf(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def is_nan(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.is_nan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.isnan(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def is_non_decreasing(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.is_non_decreasing` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.is_non_decreasing(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def is_strictly_increasing(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.is_strictly_increasing` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.is_strictly_increasing(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def l2_normalize(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.l2_normalize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(args[0] if args else 0.0)

    @staticmethod
    def lbeta(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.lbeta` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.lbeta(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def log_sigmoid(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.log_sigmoid` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _k_ops.log_sigmoid(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def log_softmax(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.log_softmax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _k_ops.log_softmax(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def multiply_no_nan(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.multiply_no_nan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.multiply_no_nan(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def ndtri(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.ndtri` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.ndtri(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def polygamma(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.polygamma` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.polygamma(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def polyval(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.polyval` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        _clean_kwargs(kwargs)
        res = args[0] if args else 0.0
        return _wrap_tensor(res)

    @staticmethod
    def reciprocal_no_nan(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reciprocal_no_nan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.reciprocal_no_nan(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def reduce_euclidean_norm(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_euclidean_norm` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.reduce_euclidean_norm(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def reduce_logsumexp(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.reduce_logsumexp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.reduce_logsumexp(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def rint(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.rint` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.rint(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def scalar_mul(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.scalar_mul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.scalar_mul(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def segment_max(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.segment_max` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.segment_max(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def segment_mean(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.segment_mean` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.segment_mean(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def segment_min(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.segment_min` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.segment_min(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def segment_prod(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.segment_prod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.segment_prod(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def segment_sum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.segment_sum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        _clean_kwargs(kwargs)
        res = as_operand(
            _ops.segment_sum(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else [0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def sigmoid(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Tensor:
        """Execute `math.sigmoid` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        from . import nn

        x = args[0] if args else None
        return nn.sigmoid(x)

    @staticmethod
    def sobol_sample(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.sobol_sample` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
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
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Tensor:
        """Execute `math.softmax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        from . import nn

        logits = args[0] if args else None
        axis_raw = args[1] if len(args) > 1 else kwargs.get("axis")
        axis = int(axis_raw) if isinstance(axis_raw, int) else None
        return nn.softmax(logits, axis=axis)

    @staticmethod
    def softplus(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.softplus` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _k_ops.softplus(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def softsign(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.softsign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _k_ops.softsign(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def squared_difference(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.squared_difference` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.squared_difference(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def top_k(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor] | TensorOperand:
        """Execute `math.top_k` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _k_ops.top_k(_to_tensor(args[0]), *args[1:], **clean_kw)
        return _wrap(res)

    @staticmethod
    def truediv(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.truediv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.true_divide(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def unsorted_segment_max(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_max` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_mean` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_min` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_prod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_sqrt_n` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.unsorted_segment_sum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.xdivy` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(_ops.xdivy(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs)))

    @staticmethod
    def xlog1py(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.xlog1py` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(_ops.xlog1py(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs)))

    @staticmethod
    def xlogy(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.xlogy` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.xlogy(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)

    @staticmethod
    def zero_fraction(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.zero_fraction` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        return _wrap_tensor(
            _ops.zero_fraction(*[_to_tensor(a) for a in args], **_clean_kwargs(kwargs))
        )

    @staticmethod
    def zeta(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `math.zeta` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        clean_kw = _clean_kwargs(kwargs)
        res = _ops.zeta(*[_to_tensor(a) for a in args], **clean_kw)
        return _wrap_tensor(res)


def abs(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `abs` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.abs(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def acos(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `acos` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.acos(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def acosh(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `acosh` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.acosh(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def add(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `add` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.add(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def reduce_all(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `reduce_all` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.all(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def reduce_any(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `reduce_any` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.any(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def range(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `range` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.arange(*args, **clean_kw)
    return _wrap_tensor(res)


def argmax(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `argmax` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.argmax(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def argmin(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `argmin` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.argmin(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def asin(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `asin` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.asin(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def asinh(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `asinh` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.asinh(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def atan(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `atan` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.atan(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def atan2(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `atan2` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.atan2(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def atanh(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `atanh` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.atanh(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def bitcast(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `bitcast` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.bitcast(_to_tensor(args[0]), **clean_kw)
    return _wrap_tensor(res)


def broadcast_to(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `broadcast_to` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.broadcast_to(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def cast(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `cast` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.cast(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def concat(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `concat` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    first = (
        [_to_tensor(item) for item in args[0]]
        if args and isinstance(args[0], (builtins.list, builtins.tuple))
        else [_to_tensor(args[0])]
    )
    res = _k_ops.concatenate(first, *args[1:], **clean_kw)
    return _wrap_tensor(res)


def cos(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `cos` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.cos(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def cosh(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `cosh` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.cosh(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def divide(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `divide` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.divide(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def einsum(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `einsum` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    _clean_kwargs(kwargs)
    res = _k_ops.einsum(
        *[_to_tensor(a) if not isinstance(a, str) else a for a in args],
        **kwargs,
    )
    return _wrap_tensor(res)


def equal(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `equal` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.equal(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def exp(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `exp` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.exp(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def expand_dims(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `expand_dims` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.expand_dims(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def eye(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `eye` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.eye(*args, **clean_kw)
    return _wrap_tensor(res)


def reshape(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `reshape` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.reshape(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def pow(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `pow` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.power(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def floor(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `floor` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.floor(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def fill(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `fill` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.full(*args, **clean_kw)
    return _wrap_tensor(res)


def gather(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `gather` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.take(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def gather_nd(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `gather_nd` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.take(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def greater(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `greater` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.greater(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def greater_equal(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `greater_equal` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.greater_equal(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def identity(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `identity` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    del kwargs
    res = _to_tensor(args[0]) if args else _to_tensor(0.0)
    return _wrap_tensor(res)


def less(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `less` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.less(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def less_equal(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `less_equal` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.less_equal(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def linspace(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `linspace` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.linspace(*args, **clean_kw)
    return _wrap_tensor(res)


def logical_and(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `logical_and` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.logical_and(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def logical_not(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `logical_not` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.logical_not(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def logical_or(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `logical_or` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.logical_or(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def matmul(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `matmul` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.matmul(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def reduce_max(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `reduce_max` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.max(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def maximum(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `maximum` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.maximum(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def reduce_mean(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `reduce_mean` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.mean(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def meshgrid(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
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


def reduce_min(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `reduce_min` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.min(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def minimum(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `minimum` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.minimum(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def multiply(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `multiply` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.multiply(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def negative(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `negative` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.negative(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def norm(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `norm` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.norm(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def not_equal(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `not_equal` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.not_equal(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def ones(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `ones` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.ones(*args, **clean_kw)
    return _wrap_tensor(res)


def ones_like(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `ones_like` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.ones_like(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def reduce_prod(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `reduce_prod` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.prod(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def repeat(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `repeat` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.repeat(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def roll(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `roll` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.roll(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def round(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `round` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.round(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def scatter_nd(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `scatter_nd` operation.

    Args:
        *args: Positional operation arguments (`indices`, `updates`, `shape`).
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    del kwargs
    updates_t = _to_tensor(args[1] if len(args) > 1 else 0.0)
    shape_arg = args[2] if len(args) > 2 else updates_t.shape
    base = _k_ops.zeros(shape_arg, dtype=updates_t.dtype)
    idx_t = _to_tensor(args[0] if args else [0])
    res = _k_ops.scatter(base, 0, idx_t, updates_t)
    return _wrap_tensor(res)


def shape(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> builtins.tuple[Tensor, ...]:
    """Execute `shape` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        tuple[Tensor, ...]: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.shape(*[_to_tensor(a) for a in args], **clean_kw)
    wrapped = _wrap(res)
    return (
        wrapped
        if isinstance(wrapped, builtins.tuple)
        else (wrapped if isinstance(wrapped, Tensor) else _wrap_tensor(wrapped),)
    )


def sign(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `sign` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.sign(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def sin(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `sin` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.sin(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def sinh(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `sinh` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.sinh(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def slice(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `slice` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
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
        *[_to_tensor(a) if i == 0 else a for i, a in enumerate(args)], **_clean_kwargs(kwargs)
    )
    return _wrap_tensor(res)


def split(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> builtins.list[Tensor]:
    """Execute `split` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        list[Tensor]: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.split(_to_tensor(args[0]), *args[1:], **clean_kw)
    wrapped = _wrap(res)
    return (
        builtins.list(wrapped)
        if isinstance(wrapped, (builtins.list, builtins.tuple))
        else [wrapped]
    )


def sqrt(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `sqrt` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.sqrt(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def square(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `square` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.square(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def squeeze(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `squeeze` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.squeeze(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def stack(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `stack` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    first = (
        [_to_tensor(item) for item in args[0]]
        if args and isinstance(args[0], (builtins.list, builtins.tuple))
        else [_to_tensor(args[0])]
    )
    res = _k_ops.stack(first, *args[1:], **clean_kw)
    return _wrap_tensor(res)


def strided_slice(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `strided_slice` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
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


def subtract(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `subtract` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.subtract(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def reduce_sum(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `reduce_sum` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.sum(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def tan(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `tan` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.tan(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def tanh(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `tanh` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _ops.tanh(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def tile(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `tile` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.tile(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def transpose(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `transpose` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.transpose(_to_tensor(args[0]), *args[1:], **clean_kw)
    return _wrap_tensor(res)


def unstack(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> builtins.list[Tensor]:
    """Execute `unstack` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        list[Tensor]: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.unstack(_to_tensor(args[0]), *args[1:], **clean_kw)
    wrapped = _wrap(res)
    return (
        builtins.list(wrapped)
        if isinstance(wrapped, (builtins.list, builtins.tuple))
        else [wrapped]
    )


def where(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `where` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.where(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def zeros(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `zeros` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.zeros(*args, **clean_kw)
    return _wrap_tensor(res)


def zeros_like(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `zeros_like` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.zeros_like(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)


def tensordot(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor:
    """Execute `tensordot` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor: Operation result.
    """
    clean_kw = _clean_kwargs(kwargs)
    res = _k_ops.tensordot(*[_to_tensor(a) for a in args], **clean_kw)
    return _wrap_tensor(res)
