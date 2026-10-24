"""TensorFlow `linalg` linear algebra operations module."""

from __future__ import annotations

import builtins

from zero_tensorflow._typing import TensorOperand, as_operand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap, _wrap_tensor

import ml_switcheroo_compiler.ops as _msc_ops

from zero_keras import ops as _ops  # isort: skip

__all__ = ["linalg"]


class linalg:
    """TensorFlow `linalg` namespace."""

    @staticmethod
    def tensordot(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.tensordot` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.tensordot(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap_tensor(res)

    @staticmethod
    def cholesky(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.cholesky` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.cholesky(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap_tensor(res)

    @staticmethod
    def det(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.det` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.det(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap_tensor(res)

    @staticmethod
    def diag(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.diag` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.diag(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap_tensor(res)

    @staticmethod
    def eigh(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
        """Execute `linalg.eigh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | tuple[Tensor, ...] | list[Tensor]: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.eigh(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def eigvalsh(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.eigvalsh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        eigh_res = _ops.eigh(*[_to_tensor(a) for a in args], **kwargs)
        vals = eigh_res[0] if isinstance(eigh_res, builtins.tuple) and eigh_res else eigh_res
        return _wrap_tensor(vals)

    @staticmethod
    def einsum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.einsum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.einsum(
            *[_to_tensor(a) if not isinstance(a, str) else a for a in args],
            **kwargs,
        )
        return _wrap_tensor(res)

    @staticmethod
    def eye(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.eye` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.eye(*args, **kwargs)
        return _wrap_tensor(res)

    @staticmethod
    def inv(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.inv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.inv(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap_tensor(res)

    @staticmethod
    def lstsq(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.lstsq` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        del kwargs
        res = as_operand(
            _msc_ops.lstsq(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else args[0]))
        )
        return _wrap_tensor(res)

    @staticmethod
    def matmul(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.matmul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.matmul(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap_tensor(res)

    @staticmethod
    def matrix_transpose(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.matrix_transpose` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        del kwargs
        res = as_operand(_msc_ops.matrix_transpose(_to_tensor(args[0])))
        return _wrap_tensor(res)

    @staticmethod
    def norm(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.norm` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.norm(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap_tensor(res)

    @staticmethod
    def pinv(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.pinv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.pinv(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap_tensor(res)

    @staticmethod
    def qr(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
        """Execute `linalg.qr` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | tuple[Tensor, ...] | list[Tensor]: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.qr(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def slogdet(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
        """Execute `linalg.slogdet` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | tuple[Tensor, ...] | list[Tensor]: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.slogdet(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def solve(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `linalg.solve` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        del kwargs
        res = _ops.solve(_to_tensor(args[0]), _to_tensor(args[1] if len(args) > 1 else args[0]))
        return _wrap_tensor(res)

    @staticmethod
    def svd(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
        """Execute `linalg.svd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | tuple[Tensor, ...] | list[Tensor]: Operation result.
        """
        kwargs = {"axis" if k == "dim" else k: v for (k, v) in kwargs.items()}
        res = _ops.svd(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)
