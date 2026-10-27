"""TensorFlow `linalg` linear algebra operations module."""

from __future__ import annotations

import builtins
import itertools
import math
from collections.abc import Sequence

from zero_tensorflow._typing import DTypeLike, IntTuple, TensorOperand, as_operand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap, _wrap_tensor, to_dtype
from zero_tensorflow.dtypes import DType

import ml_switcheroo_compiler.ops as _msc_ops
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer

from zero_keras import ops as _ops  # isort: skip

__all__ = [
    "LinearOperator",
    "LinearOperatorAdjoint",
    "LinearOperatorBlockDiag",
    "LinearOperatorBlockLowerTriangular",
    "LinearOperatorCirculant",
    "LinearOperatorCirculant2D",
    "LinearOperatorCirculant3D",
    "LinearOperatorComposition",
    "LinearOperatorDiag",
    "LinearOperatorFullMatrix",
    "LinearOperatorHouseholder",
    "LinearOperatorIdentity",
    "LinearOperatorInversion",
    "LinearOperatorKronecker",
    "LinearOperatorLowRankUpdate",
    "LinearOperatorLowerTriangular",
    "LinearOperatorPermutation",
    "LinearOperatorScaledIdentity",
    "LinearOperatorToeplitz",
    "LinearOperatorTridiag",
    "LinearOperatorZeros",
    "linalg",
]


def _to_nested_floats(t: Tensor) -> list[list[float]]:
    """Extract a 2-D matrix `Tensor` into a nested list of floats.

    Args:
        t: 2-D matrix `Tensor`.

    Returns:
        list[list[float]]: 2-D list of floats.
    """
    raw = t.tolist()
    if not isinstance(raw, builtins.list):
        return [[float(raw) if isinstance(raw, (int, float)) else 0.0]]
    return [
        [float(c) if isinstance(c, (int, float)) else 0.0 for c in r]
        if isinstance(r, builtins.list)
        else [float(r) if isinstance(r, (int, float)) else 0.0]
        for r in raw
    ]


def _to_1d_floats(t: Tensor) -> list[float]:
    """Extract a 1-D vector `Tensor` into a list of floats.

    Args:
        t: 1-D vector `Tensor`.

    Returns:
        list[float]: 1-D list of floats.
    """
    raw = t.tolist()
    return (
        [float(x) if isinstance(x, (int, float)) else 0.0 for x in raw]
        if isinstance(raw, builtins.list)
        else [float(raw) if isinstance(raw, (int, float)) else 0.0]
    )


def _iter_batch_indices(batch_shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Generate all multi-indices for `batch_shape`.

    Args:
        batch_shape: Leading batch dimensions tuple.

    Returns:
        list[tuple[int, ...]]: List of multi-index tuples.
    """
    return list(itertools.product(*(range(d) for d in batch_shape)))


def _slice_batch_matrix(t: Tensor, idx: tuple[int, ...]) -> Tensor:
    """Slice a single 2-D matrix from a batched tensor `t` at `idx`.

    Args:
        t: Batched matrix tensor of shape `[*batch_shape, M, N]`.
        idx: Batch index tuple of length `len(batch_shape)`.

    Returns:
        Tensor: 2-D matrix slice `Tensor`.
    """
    cur = t
    for i in idx:
        cur = cur[i]
    return cur


def _stack_batch_results(
    items: list[Tensor],
    batch_shape: tuple[int, ...],
    inner_shape: tuple[int, ...],
    dtype: DType | None,
) -> Tensor:
    """Stack flat batch results back into `[*batch_shape, *inner_shape]`.

    Args:
        items: List of result tensors for each batch index.
        batch_shape: Leading batch dimensions tuple.
        inner_shape: Inner matrix/vector shape tuple.
        dtype: Output tensor `DType`.

    Returns:
        Tensor: Reshaped batched `Tensor`.
    """
    stacked = _ops.stack([_to_tensor(it) for it in items], axis=0)
    full_shape = batch_shape + inner_shape
    dt_str = dtype.value if dtype is not None else "float32"
    return _wrap_tensor(_ops.cast(_ops.reshape(stacked, full_shape), dt_str))


class LinearOperator:
    """Base class defining a [batch of] linear operator[s]."""

    def __init__(
        self,
        dtype: DTypeLike = None,
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = True,
        name: str = "LinearOperator",
        matrix: Tensor | TensorOperand = None,
    ) -> None:
        """Initialize `LinearOperator`.

        Args:
            dtype: Data type of the linear operator.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the linear operator.
            matrix: Optional underlying dense matrix tensor.
        """
        self._matrix = (
            Tensor([[1.0]], dtype=dtype) if matrix is None else Tensor(matrix, dtype=dtype)
        )
        self._dtype = self._matrix.dtype or to_dtype(dtype)
        self.is_non_singular = is_non_singular
        self.is_self_adjoint = is_self_adjoint
        self.is_positive_definite = is_positive_definite
        self.is_square = is_square
        self.name = name

    @property
    def dtype(self) -> DType:
        """Return the `DType` of the operator.

        Returns:
            DType: Operator element data type.
        """
        return self._dtype

    @property
    def shape(self) -> IntTuple:
        """Return the static shape tuple of the operator.

        Returns:
            tuple[int, ...]: Operator dimensions.
        """
        return self._to_dense().shape

    def _to_dense(self) -> Tensor:
        """Construct the dense matrix representation of this operator.

        Returns:
            Tensor: Dense matrix tensor.
        """
        return self._matrix

    def to_dense(self, name: str = "to_dense") -> Tensor:
        """Convert this `LinearOperator` into a dense `Tensor`.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Dense matrix representation.
        """
        del name
        return self._to_dense()

    def _matmul(
        self,
        x: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Specialized matrix multiplication implementation `self @ x`.

        Args:
            x: Right-hand matrix `Tensor`.
            adjoint: If `True`, multiply by the adjoint of `self`.
            adjoint_arg: If `True`, multiply by the adjoint of `x`.

        Returns:
            Tensor: Matrix product tensor.
        """
        lhs = linalg.adjoint(self._to_dense()) if adjoint else self._to_dense()
        rhs = linalg.adjoint(x) if adjoint_arg else x
        return linalg.matmul(lhs, rhs)

    def matmul(
        self,
        x: Tensor | TensorOperand,
        adjoint: bool = False,
        adjoint_arg: bool = False,
        name: str = "matmul",
    ) -> Tensor:
        """Transform [batch] matrix `x` with left multiplication `self @ x`.

        Args:
            x: Right-hand matrix operand.
            adjoint: If `True`, multiply by the adjoint of `self`.
            adjoint_arg: If `True`, multiply by the adjoint of `x`.
            name: Optional operation name.

        Returns:
            Tensor: Matrix product tensor.
        """
        del name
        return self._matmul(Tensor(x, dtype=self.dtype), adjoint=adjoint, adjoint_arg=adjoint_arg)

    def matvec(
        self,
        x: Tensor | TensorOperand,
        adjoint: bool = False,
        name: str = "matvec",
    ) -> Tensor:
        """Transform [batch] vector `x` with left multiplication `self @ x`.

        Args:
            x: Right-hand vector operand.
            adjoint: If `True`, multiply by the adjoint of `self`.
            name: Optional operation name.

        Returns:
            Tensor: Matrix-vector product tensor.
        """
        del name
        x_col = _wrap_tensor(_ops.expand_dims(_to_tensor(Tensor(x, dtype=self.dtype)), -1))
        prod = self._matmul(x_col, adjoint=adjoint, adjoint_arg=False)
        return _wrap_tensor(_ops.squeeze(_to_tensor(prod), axis=-1))

    def _solve(
        self,
        rhs: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Specialized linear system solver `self @ X = rhs`.

        Args:
            rhs: Right-hand side matrix `Tensor`.
            adjoint: If `True`, solve with the adjoint of `self`.
            adjoint_arg: If `True`, solve with the adjoint of `rhs`.

        Returns:
            Tensor: Solution tensor `X`.
        """
        mat = linalg.adjoint(self._to_dense()) if adjoint else self._to_dense()
        b = linalg.adjoint(rhs) if adjoint_arg else rhs
        return linalg.solve(mat, b)

    def solve(
        self,
        rhs: Tensor | TensorOperand,
        adjoint: bool = False,
        adjoint_arg: bool = False,
        name: str = "solve",
    ) -> Tensor:
        """Solve `self @ X = rhs` for `X`.

        Args:
            rhs: Right-hand side matrix tensor.
            adjoint: If `True`, solve with the adjoint of `self`.
            adjoint_arg: If `True`, solve with the adjoint of `rhs`.
            name: Optional operation name.

        Returns:
            Tensor: Solution tensor `X`.
        """
        del name
        return self._solve(
            Tensor(rhs, dtype=self.dtype),
            adjoint=adjoint,
            adjoint_arg=adjoint_arg,
        )

    def _determinant(self) -> Tensor:
        """Specialized determinant computation.

        Returns:
            Tensor: Determinant tensor.
        """
        return linalg.det(self._to_dense())

    def determinant(self, name: str = "det") -> Tensor:
        """Compute the determinant for every batch member.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Determinant tensor.
        """
        del name
        return self._determinant()

    def _log_abs_determinant(self) -> Tensor:
        """Specialized log-abs-determinant computation.

        Returns:
            Tensor: Log-abs-determinant tensor.
        """
        det_t = _to_tensor(self._determinant())
        return _wrap_tensor(_msc_ops.log(_msc_ops.abs(det_t)))

    def log_abs_determinant(self, name: str = "log_abs_det") -> Tensor:
        """Compute the log of the absolute value of the determinant.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Log-abs-determinant tensor.
        """
        del name
        return self._log_abs_determinant()

    def _diag_part(self) -> Tensor:
        """Specialized diagonal part extraction.

        Returns:
            Tensor: Diagonal vector tensor.
        """
        return linalg.diag_part(self._to_dense())

    def diag_part(self, name: str = "diag_part") -> Tensor:
        """Extract the main diagonal of this linear operator.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Diagonal vector tensor.
        """
        del name
        return self._diag_part()

    def _trace(self) -> Tensor:
        """Specialized trace computation.

        Returns:
            Tensor: Trace tensor.
        """
        return _wrap_tensor(_ops.sum(_to_tensor(self._diag_part()), axis=-1))

    def trace(self, name: str = "trace") -> Tensor:
        """Compute the trace of the linear operator.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Trace tensor.
        """
        del name
        return self._trace()

    def inverse(self, name: str = "inverse") -> LinearOperatorFullMatrix:
        """Return the `LinearOperator` representing the inverse of this operator.

        Args:
            name: Optional operation name.

        Returns:
            LinearOperatorFullMatrix: Inverse linear operator.
        """
        return LinearOperatorFullMatrix(linalg.inv(self._to_dense()), name=name)

    def cholesky(self, name: str = "cholesky") -> LinearOperatorLowerTriangular:
        """Return a Cholesky factor as a `LinearOperatorLowerTriangular`.

        Args:
            name: Optional operation name.

        Returns:
            LinearOperatorLowerTriangular: Lower-triangular Cholesky factor operator.
        """
        return LinearOperatorLowerTriangular(linalg.cholesky(self._to_dense()), name=name)

    def adjoint(self, name: str = "adjoint") -> LinearOperatorFullMatrix:
        """Return the conjugate transpose (adjoint) `LinearOperator`.

        Args:
            name: Optional operation name.

        Returns:
            LinearOperatorFullMatrix: Adjoint linear operator.
        """
        return LinearOperatorFullMatrix(linalg.adjoint(self._to_dense()), name=name)

    @property
    def H(self) -> LinearOperatorFullMatrix:
        """Return the conjugate transpose (adjoint) `LinearOperator` (`self.adjoint()`).

        Returns:
            LinearOperatorFullMatrix: Adjoint linear operator.
        """
        return self.adjoint()

    @property
    def graph_parents(self) -> builtins.list[Tensor]:
        """Return the list of graph parents of this `LinearOperator`.

        Returns:
            list[Tensor]: Underlying matrix tensor list.
        """
        return [self._matrix]

    @property
    def parameters(self) -> dict[str, Tensor | TensorOperand]:
        """Return the dictionary of parameters used to instantiate this `LinearOperator`.

        Returns:
            dict[str, Tensor | TensorOperand]: Parameter dictionary.
        """
        return {"name": self.name}


class LinearOperatorFullMatrix(LinearOperator):
    """`LinearOperator` acting like a [batch] square or non-square dense matrix."""

    def __init__(
        self,
        matrix: Tensor | TensorOperand = ((1.0,),),
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = None,
        name: str = "LinearOperatorFullMatrix",
    ) -> None:
        """Initialize `LinearOperatorFullMatrix`.

        Args:
            matrix: Dense matrix tensor of shape `[..., M, N]`.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        t_mat = Tensor(matrix)
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )


class LinearOperatorDiag(LinearOperator):
    """`LinearOperator` acting like a [batch] square diagonal matrix."""

    def __init__(
        self,
        diag: Tensor | TensorOperand = (1.0,),
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = True,
        name: str = "LinearOperatorDiag",
    ) -> None:
        """Initialize `LinearOperatorDiag`.

        Args:
            diag: Diagonal vector tensor of shape `[..., N]`.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        self.diag = Tensor(diag)
        t_mat = linalg.diag(self.diag)
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Construct dense diagonal matrix from `self.diag`.

        Returns:
            Tensor: Dense diagonal matrix tensor.
        """
        return linalg.diag(self.diag)

    def _diag_part(self) -> Tensor:
        """Return `self.diag` directly.

        Returns:
            Tensor: Diagonal vector tensor.
        """
        return self.diag

    def _trace(self) -> Tensor:
        """Compute trace as the sum of `self.diag` along the last axis.

        Returns:
            Tensor: Trace tensor.
        """
        return _wrap_tensor(_ops.sum(_to_tensor(self.diag), axis=-1))

    def _determinant(self) -> Tensor:
        """Compute determinant as the product of `self.diag` along the last axis.

        Returns:
            Tensor: Determinant tensor.
        """
        return _wrap_tensor(_ops.prod(_to_tensor(self.diag), axis=-1))

    def _log_abs_determinant(self) -> Tensor:
        """Compute log-abs-determinant as the sum of `log(abs(self.diag))`.

        Returns:
            Tensor: Log-abs-determinant tensor.
        """
        return _wrap_tensor(_ops.sum(_msc_ops.log(_msc_ops.abs(_to_tensor(self.diag))), axis=-1))

    def _matmul(
        self,
        x: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Multiply diagonal vector elementwise with rows of `x`.

        Args:
            x: Right-hand matrix `Tensor`.
            adjoint: If `True`, conjugate the diagonal.
            adjoint_arg: If `True`, multiply by the adjoint of `x`.

        Returns:
            Tensor: Matrix product tensor.
        """
        del adjoint
        rhs = linalg.adjoint(x) if adjoint_arg else x
        d_col = _ops.expand_dims(_to_tensor(self.diag), -1)
        return _wrap_tensor(_msc_ops.multiply(d_col, _to_tensor(rhs)))

    def _solve(
        self,
        rhs: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Solve diagonal system by dividing rows of `rhs` by `self.diag`.

        Args:
            rhs: Right-hand side matrix `Tensor`.
            adjoint: If `True`, conjugate the diagonal.
            adjoint_arg: If `True`, solve with the adjoint of `rhs`.

        Returns:
            Tensor: Solution tensor.
        """
        del adjoint
        b = linalg.adjoint(rhs) if adjoint_arg else rhs
        d_col = _ops.expand_dims(_to_tensor(self.diag), -1)
        return _wrap_tensor(_msc_ops.divide(_to_tensor(b), d_col))


class LinearOperatorIdentity(LinearOperator):
    """`LinearOperator` acting like a [batch] square identity matrix."""

    def __init__(
        self,
        num_rows: int = 1,
        batch_shape: IntTuple | None = None,
        dtype: DTypeLike = None,
        is_non_singular: bool = True,
        is_self_adjoint: bool = True,
        is_positive_definite: bool = True,
        is_square: bool = True,
        assert_proper_shapes: bool = False,
        name: str = "LinearOperatorIdentity",
    ) -> None:
        """Initialize `LinearOperatorIdentity`.

        Args:
            num_rows: Number of rows and columns in the identity matrix.
            batch_shape: Optional leading batch dimensions.
            dtype: Data type of the identity matrix.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            assert_proper_shapes: Whether to assert proper shapes.
            name: Name of the operator.
        """
        del assert_proper_shapes
        self._num_rows = int(num_rows)
        self._batch_shape: IntTuple = tuple(batch_shape) if batch_shape is not None else ()
        resolved_dtype = to_dtype(dtype)
        t_mat = linalg.eye(
            self._num_rows,
            batch_shape=self._batch_shape if self._batch_shape else None,
            dtype=resolved_dtype.value,
        )
        super().__init__(
            dtype=resolved_dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return the identity matrix tensor.

        Returns:
            Tensor: Identity matrix tensor.
        """
        return self._matrix

    def _diag_part(self) -> Tensor:
        """Return ones vector of length `num_rows`.

        Returns:
            Tensor: Ones diagonal vector.
        """
        return _wrap_tensor(_ops.ones((*self._batch_shape, self._num_rows), dtype=self.dtype.value))

    def _trace(self) -> Tensor:
        """Return `num_rows` as trace.

        Returns:
            Tensor: Trace tensor.
        """
        return _wrap_tensor(
            _ops.full(self._batch_shape, float(self._num_rows), dtype=self.dtype.value)
        )

    def _determinant(self) -> Tensor:
        """Return `1` as determinant.

        Returns:
            Tensor: Determinant tensor.
        """
        return _wrap_tensor(_ops.ones(self._batch_shape, dtype=self.dtype.value))

    def _log_abs_determinant(self) -> Tensor:
        """Return `0` as log-abs-determinant.

        Returns:
            Tensor: Log-abs-determinant tensor.
        """
        return _wrap_tensor(_ops.zeros(self._batch_shape, dtype=self.dtype.value))

    def _matmul(
        self,
        x: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Multiply by identity (returning `x` or `adjoint(x)`).

        Args:
            x: Right-hand matrix `Tensor`.
            adjoint: Unused for identity.
            adjoint_arg: If `True`, return the adjoint of `x`.

        Returns:
            Tensor: Result tensor.
        """
        del adjoint
        return linalg.adjoint(x) if adjoint_arg else x

    def _solve(
        self,
        rhs: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Solve identity system (returning `rhs` or `adjoint(rhs)`).

        Args:
            rhs: Right-hand side `Tensor`.
            adjoint: Unused for identity.
            adjoint_arg: If `True`, return the adjoint of `rhs`.

        Returns:
            Tensor: Solution tensor.
        """
        del adjoint
        return linalg.adjoint(rhs) if adjoint_arg else rhs


class LinearOperatorLowerTriangular(LinearOperator):
    """`LinearOperator` acting like a [batch] square lower triangular matrix."""

    def __init__(
        self,
        tril: Tensor | TensorOperand = ((1.0,),),
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = True,
        name: str = "LinearOperatorLowerTriangular",
    ) -> None:
        """Initialize `LinearOperatorLowerTriangular`.

        Args:
            tril: Matrix tensor whose lower triangle defines the operator.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        self._tril = linalg.band_part(tril, -1, 0)
        super().__init__(
            dtype=self._tril.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=self._tril,
        )

    def _to_dense(self) -> Tensor:
        """Return the lower-triangular dense matrix.

        Returns:
            Tensor: Lower-triangular matrix tensor.
        """
        return self._tril

    def _diag_part(self) -> Tensor:
        """Extract main diagonal of the lower-triangular matrix.

        Returns:
            Tensor: Diagonal vector tensor.
        """
        return linalg.diag_part(self._tril)

    def _determinant(self) -> Tensor:
        """Compute determinant as product of the diagonal entries.

        Returns:
            Tensor: Determinant tensor.
        """
        return _wrap_tensor(_ops.prod(_to_tensor(self._diag_part()), axis=-1))

    def _log_abs_determinant(self) -> Tensor:
        """Compute log-abs-determinant from the diagonal entries.

        Returns:
            Tensor: Log-abs-determinant tensor.
        """
        return _wrap_tensor(
            _ops.sum(_msc_ops.log(_msc_ops.abs(_to_tensor(self._diag_part()))), axis=-1)
        )

    def _solve(
        self,
        rhs: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Solve triangular system using `linalg.triangular_solve`.

        Args:
            rhs: Right-hand side `Tensor`.
            adjoint: If `True`, solve with adjoint of `self`.
            adjoint_arg: If `True`, solve with adjoint of `rhs`.

        Returns:
            Tensor: Solution tensor.
        """
        b = linalg.adjoint(rhs) if adjoint_arg else rhs
        return linalg.triangular_solve(self._tril, b, lower=True, adjoint=adjoint)


def _extract_operator_dense(op: LinearOperator | Tensor | TensorOperand) -> Tensor:
    """Extract a dense `Tensor` from a `LinearOperator` or tensor-like operand.

    Args:
        op: `LinearOperator` or tensor operand.

    Returns:
        Tensor: Dense matrix tensor.
    """
    return op.to_dense() if isinstance(op, LinearOperator) else Tensor(op)


class LinearOperatorAdjoint(LinearOperator):
    """`LinearOperator` representing the conjugate transpose of another operator."""

    def __init__(
        self,
        operator: LinearOperator | Tensor | TensorOperand = ((1.0,),),
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize `LinearOperatorAdjoint`.

        Args:
            operator: Underlying `LinearOperator` or matrix.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Optional name for the operator.
        """
        self.operator = operator
        t_mat = linalg.adjoint(_extract_operator_dense(operator))
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name or "LinearOperatorAdjoint",
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return the adjoint of the underlying operator's dense matrix.

        Returns:
            Tensor: Dense adjoint matrix tensor.
        """
        return linalg.adjoint(_extract_operator_dense(self.operator))

    def _matmul(
        self,
        x: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Delegate `_matmul` with flipped `adjoint` flag when `operator` is a `LinearOperator`.

        Args:
            x: Right-hand matrix `Tensor`.
            adjoint: If `True`, multiply by the adjoint of `self` (i.e. `self.operator`).
            adjoint_arg: If `True`, multiply by the adjoint of `x`.

        Returns:
            Tensor: Matrix product tensor.
        """
        if isinstance(self.operator, LinearOperator):
            return self.operator._matmul(x, adjoint=not adjoint, adjoint_arg=adjoint_arg)
        return super()._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _solve(
        self,
        rhs: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Delegate `_solve` with flipped `adjoint` flag when `operator` is a `LinearOperator`.

        Args:
            rhs: Right-hand side `Tensor`.
            adjoint: If `True`, solve with the adjoint of `self`.
            adjoint_arg: If `True`, solve with the adjoint of `rhs`.

        Returns:
            Tensor: Solution tensor.
        """
        if isinstance(self.operator, LinearOperator):
            return self.operator._solve(rhs, adjoint=not adjoint, adjoint_arg=adjoint_arg)
        return super()._solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)


def _build_block_diag_dense(mats: list[Tensor]) -> Tensor:
    """Construct a block-diagonal dense matrix from a list of 2-D+ matrix `Tensor`s.

    Args:
        mats: Non-empty list of matrix `Tensor`s.

    Returns:
        Tensor: Block-diagonal matrix `Tensor`.
    """
    if not mats:
        return Tensor([[1.0]])
    if len(mats) == 1:
        return mats[0]
    dt = mats[0].dtype or DType.Float32
    total_rows = sum(m.shape[-2] if len(m.shape) >= 2 else 1 for m in mats)
    total_cols = sum(m.shape[-1] if len(m.shape) >= 2 else 1 for m in mats)
    grid: list[list[float]] = [[0.0 for _ in range(total_cols)] for _ in range(total_rows)]
    r_off = 0
    c_off = 0
    for m in mats:
        sub = _to_nested_floats(m)
        r_len = len(sub)
        c_len = len(sub[0]) if sub else 0
        for i in range(r_len):
            for j in range(c_len):
                grid[r_off + i][c_off + j] = sub[i][j]
        r_off += r_len
        c_off += c_len
    return Tensor(grid, dtype=dt)


class LinearOperatorBlockDiag(LinearOperator):
    """`LinearOperator` combining one or more operators into a block-diagonal matrix."""

    def __init__(
        self,
        operators: Sequence[LinearOperator | Tensor | TensorOperand] = (((1.0,),),),
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool = True,
        name: str | None = None,
    ) -> None:
        """Initialize `LinearOperatorBlockDiag`.

        Args:
            operators: Sequence of `LinearOperator` or matrix tensors.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Optional name for the operator.
        """
        self.operators = list(operators)
        mats = [_extract_operator_dense(op) for op in self.operators]
        t_mat = _build_block_diag_dense(mats)
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name or "LinearOperatorBlockDiag",
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Construct the block-diagonal dense matrix from `self.operators`.

        Returns:
            Tensor: Block-diagonal dense matrix.
        """
        return _build_block_diag_dense([_extract_operator_dense(op) for op in self.operators])

    def _determinant(self) -> Tensor:
        """Compute determinant as the product of block determinants.

        Returns:
            Tensor: Determinant tensor.
        """
        if not self.operators:
            return Tensor(1.0, dtype=self.dtype)
        dets = [
            op.determinant()
            if isinstance(op, LinearOperator)
            else linalg.det(_extract_operator_dense(op))
            for op in self.operators
        ]
        acc = _to_tensor(dets[0])
        for d in dets[1:]:
            acc = _msc_ops.multiply(acc, _to_tensor(d))
        return _wrap_tensor(acc)

    def _trace(self) -> Tensor:
        """Compute trace as the sum of block traces.

        Returns:
            Tensor: Trace tensor.
        """
        if not self.operators:
            return Tensor(1.0, dtype=self.dtype)
        traces = [
            op.trace()
            if isinstance(op, LinearOperator)
            else linalg.trace(_extract_operator_dense(op))
            for op in self.operators
        ]
        acc = _to_tensor(traces[0])
        for tr in traces[1:]:
            acc = _msc_ops.add(acc, _to_tensor(tr))
        return _wrap_tensor(acc)


def _build_block_lower_triangular_dense(
    operators: Sequence[Sequence[LinearOperator | Tensor | TensorOperand]] | TensorOperand,
) -> Tensor:
    """Build a dense block-lower-triangular matrix from nested row sequences.

    Args:
        operators: Nested sequence of lower-triangular block rows.

    Returns:
        Tensor: Dense block-lower-triangular matrix `Tensor`.
    """
    if not isinstance(operators, Sequence) or isinstance(operators, (str, bytes)) or not operators:
        return Tensor([[1.0]])
    rows_ops: list[list[Tensor]] = [
        [_extract_operator_dense(op) for op in row]
        for row in operators
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) > 0
    ]
    if not rows_ops:
        return Tensor([[1.0]])
    row_heights = [r[-1].shape[-2] if len(r[-1].shape) >= 2 else 1 for r in rows_ops]
    col_widths = [r[-1].shape[-1] if len(r[-1].shape) >= 2 else 1 for r in rows_ops]
    total_r = sum(row_heights)
    total_c = sum(col_widths)
    dt = rows_ops[0][0].dtype or DType.Float32
    grid: list[list[float]] = [[0.0 for _ in range(total_c)] for _ in range(total_r)]
    r_off = 0
    for r_idx, row_mats in enumerate(rows_ops):
        c_off = 0
        for c_idx, mat in enumerate(row_mats):
            sub = _to_nested_floats(mat)
            for i, sub_row in enumerate(sub):
                for j, val in enumerate(sub_row):
                    grid[r_off + i][c_off + j] = val
            c_off += col_widths[c_idx]
        r_off += row_heights[r_idx]
    return Tensor(grid, dtype=dt)


class LinearOperatorBlockLowerTriangular(LinearOperator):
    """`LinearOperator` formed from a block-lower-triangular arrangement of operators."""

    def __init__(
        self,
        operators: Sequence[Sequence[LinearOperator | Tensor | TensorOperand]]
        | TensorOperand = None,
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = None,
        name: str = "LinearOperatorBlockLowerTriangular",
    ) -> None:
        """Initialize `LinearOperatorBlockLowerTriangular`.

        Args:
            operators: Nested sequence of row blocks defining the lower triangle.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        self.operators = operators
        t_mat = _build_block_lower_triangular_dense(operators)
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return the block-lower-triangular dense matrix.

        Returns:
            Tensor: Dense matrix tensor.
        """
        return _build_block_lower_triangular_dense(self.operators)


def _build_circulant_dense(spectrum: Tensor) -> Tensor:
    """Build a real circulant matrix from its 1-D spectrum or first-column coefficients.

    Args:
        spectrum: Spectrum tensor.

    Returns:
        Tensor: Dense `[N, N]` circulant matrix.
    """
    vals = _to_1d_floats(_wrap_tensor(_ops.reshape(_to_tensor(spectrum), (-1,))))
    n = len(vals)
    if n == 0:
        return Tensor([[1.0]], dtype=spectrum.dtype)
    # Inverse DFT of spectrum gives the first column c_k = (1/N) sum_j spectrum_j * cos(2*pi*j*k/N)
    col: list[float] = []
    for k in range(n):
        re_sum = 0.0
        for j, s_val in enumerate(vals):
            re_sum += s_val * math.cos(2.0 * math.pi * j * k / n)
        col.append(re_sum / n)
    grid: list[list[float]] = [[col[(i - j) % n] for j in range(n)] for i in range(n)]
    return Tensor(grid, dtype=spectrum.dtype)


class LinearOperatorCirculant(LinearOperator):
    """`LinearOperator` acting like a 1-D circulant matrix defined by its spectrum."""

    def __init__(
        self,
        spectrum: Tensor | TensorOperand = (1.0,),
        input_output_dtype: DTypeLike = None,
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool = True,
        name: str = "LinearOperatorCirculant",
    ) -> None:
        """Initialize `LinearOperatorCirculant`.

        Args:
            spectrum: 1-D spectrum tensor.
            input_output_dtype: Optional dtype for inputs and outputs.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        self.spectrum = Tensor(spectrum)
        self.input_output_dtype = input_output_dtype
        t_mat = _build_circulant_dense(self.spectrum)
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return the circulant dense matrix.

        Returns:
            Tensor: Circulant matrix tensor.
        """
        return _build_circulant_dense(self.spectrum)

    def _determinant(self) -> Tensor:
        """Compute determinant as the product of the spectrum eigenvalues.

        Returns:
            Tensor: Determinant tensor.
        """
        return _wrap_tensor(_ops.prod(_to_tensor(self.spectrum), axis=-1))

    def _trace(self) -> Tensor:
        """Compute trace as the sum of the spectrum eigenvalues.

        Returns:
            Tensor: Trace tensor.
        """
        return _wrap_tensor(_ops.sum(_to_tensor(self.spectrum), axis=-1))


class LinearOperatorCirculant2D(LinearOperatorCirculant):
    """`LinearOperator` acting like a 2-D block-circulant matrix."""

    def __init__(
        self,
        spectrum: Tensor | TensorOperand = (1.0,),
        input_output_dtype: DTypeLike = None,
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool = True,
        name: str = "LinearOperatorCirculant2D",
    ) -> None:
        """Initialize `LinearOperatorCirculant2D`.

        Args:
            spectrum: 2-D spectrum tensor.
            input_output_dtype: Optional dtype for inputs and outputs.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        super().__init__(
            spectrum=spectrum,
            input_output_dtype=input_output_dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
        )


class LinearOperatorCirculant3D(LinearOperatorCirculant):
    """`LinearOperator` acting like a 3-D block-circulant matrix."""

    def __init__(
        self,
        spectrum: Tensor | TensorOperand = (1.0,),
        input_output_dtype: DTypeLike = None,
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool = True,
        name: str = "LinearOperatorCirculant3D",
    ) -> None:
        """Initialize `LinearOperatorCirculant3D`.

        Args:
            spectrum: 3-D spectrum tensor.
            input_output_dtype: Optional dtype for inputs and outputs.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        super().__init__(
            spectrum=spectrum,
            input_output_dtype=input_output_dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
        )


class LinearOperatorComposition(LinearOperator):
    """`LinearOperator` composing a sequence of operators `A_0 @ A_1 @ ...`."""

    def __init__(
        self,
        operators: Sequence[LinearOperator | Tensor | TensorOperand] = (((1.0,),),),
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize `LinearOperatorComposition`.

        Args:
            operators: Sequence of `LinearOperator` or matrix tensors to multiply.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Optional name for the operator.
        """
        self.operators = list(operators)
        t_mat = self._compute_composed_dense()
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name or "LinearOperatorComposition",
            matrix=t_mat,
        )

    def _compute_composed_dense(self) -> Tensor:
        """Multiply all constituent operator matrices left-to-right.

        Returns:
            Tensor: Composed dense matrix tensor.
        """
        mats = [_extract_operator_dense(op) for op in self.operators]
        acc = mats[0] if mats else Tensor([[1.0]])
        for m in mats[1:]:
            acc = linalg.matmul(acc, m)
        return acc

    def _to_dense(self) -> Tensor:
        """Return the composed dense matrix.

        Returns:
            Tensor: Composed dense matrix tensor.
        """
        return self._compute_composed_dense()

    def _matmul(
        self,
        x: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Apply operators sequentially from right to left (`A_0 @ (A_1 @ x)`).

        Args:
            x: Right-hand matrix `Tensor`.
            adjoint: If `True`, apply adjoints in reverse order.
            adjoint_arg: If `True`, multiply by the adjoint of `x`.

        Returns:
            Tensor: Matrix product tensor.
        """
        cur = linalg.adjoint(x) if adjoint_arg else x
        if not self.operators:
            return cur
        ops_order = list(self.operators) if adjoint else list(reversed(self.operators))
        for op in ops_order:
            if isinstance(op, LinearOperator):
                cur = op.matmul(cur, adjoint=adjoint)
            else:
                mat = (
                    linalg.adjoint(_extract_operator_dense(op))
                    if adjoint
                    else _extract_operator_dense(op)
                )
                cur = linalg.matmul(mat, cur)
        return cur


def _build_householder_dense(reflection_axis: Tensor) -> Tensor:
    """Build Householder reflection matrix `I - 2 v v^T / (v^T v)`.

    Args:
        reflection_axis: 1-D or batched reflection vector tensor.

    Returns:
        Tensor: Dense Householder reflection matrix.
    """
    n = reflection_axis.shape[-1] if reflection_axis.shape else 1
    dt_str = reflection_axis.dtype.value if reflection_axis.dtype is not None else "float32"
    eye_m = _to_tensor(linalg.eye(n, dtype=dt_str))
    if not reflection_axis.shape:
        return _wrap_tensor(_msc_ops.negative(eye_m))
    v = _to_tensor(reflection_axis)
    norm_sq = _ops.sum(_msc_ops.multiply(v, v), axis=-1, keepdims=True)
    norm_sq_safe = _ops.where(
        _msc_ops.greater(norm_sq, _to_tensor(1e-12, dtype=reflection_axis.dtype)),
        norm_sq,
        _to_tensor(1.0, dtype=reflection_axis.dtype),
    )
    v_col = _ops.expand_dims(v, -1)
    v_row = _ops.expand_dims(v, -2)
    outer = _ops.matmul(v_col, v_row)
    two_outer = _msc_ops.multiply(_to_tensor(2.0, dtype=reflection_axis.dtype), outer)
    scaled = _msc_ops.divide(two_outer, _ops.expand_dims(norm_sq_safe, -1))
    return _wrap_tensor(_msc_ops.subtract(eye_m, scaled))


class LinearOperatorHouseholder(LinearOperator):
    """`LinearOperator` acting like a Householder reflection matrix `I - 2 v v^H / ||v||^2`."""

    def __init__(
        self,
        reflection_axis: Tensor | TensorOperand = (1.0,),
        is_non_singular: bool | None = True,
        is_self_adjoint: bool | None = True,
        is_positive_definite: bool | None = False,
        is_square: bool | None = True,
        name: str = "LinearOperatorHouseholder",
    ) -> None:
        """Initialize `LinearOperatorHouseholder`.

        Args:
            reflection_axis: Reflection axis vector tensor.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        self.reflection_axis = Tensor(reflection_axis)
        t_mat = _build_householder_dense(self.reflection_axis)
        super().__init__(
            dtype=self.reflection_axis.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return the Householder reflection dense matrix.

        Returns:
            Tensor: Householder reflection matrix tensor.
        """
        return _build_householder_dense(self.reflection_axis)

    def _determinant(self) -> Tensor:
        """Return `-1.0` as the determinant of a Householder reflection.

        Returns:
            Tensor: Scalar `-1.0` tensor.
        """
        return Tensor(-1.0, dtype=self.dtype)

    def _log_abs_determinant(self) -> Tensor:
        """Return `0.0` as the log-abs-determinant of a Householder reflection.

        Returns:
            Tensor: Scalar `0.0` tensor.
        """
        return Tensor(0.0, dtype=self.dtype)

    def _solve(
        self,
        rhs: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Solve `H @ X = rhs` using `H^{-1} = H`.

        Args:
            rhs: Right-hand side `Tensor`.
            adjoint: If `True`, solve with adjoint.
            adjoint_arg: If `True`, solve with adjoint of `rhs`.

        Returns:
            Tensor: Solution tensor `H @ rhs`.
        """
        return self._matmul(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)


class LinearOperatorInversion(LinearOperator):
    """`LinearOperator` representing the inverse of another operator."""

    def __init__(
        self,
        operator: LinearOperator | Tensor | TensorOperand = ((1.0,),),
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize `LinearOperatorInversion`.

        Args:
            operator: Underlying `LinearOperator` or matrix to invert.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Optional name for the operator.
        """
        self.operator = operator
        t_mat = linalg.inv(_extract_operator_dense(operator))
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name or "LinearOperatorInversion",
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return the inverse of the underlying operator's dense matrix.

        Returns:
            Tensor: Inverse dense matrix tensor.
        """
        return linalg.inv(_extract_operator_dense(self.operator))

    def _matmul(
        self,
        x: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Multiply by `A^{-1}` via `A.solve(x)`.

        Args:
            x: Right-hand matrix `Tensor`.
            adjoint: If `True`, multiply by `(A^{-1})^H`.
            adjoint_arg: If `True`, multiply by `x^H`.

        Returns:
            Tensor: Matrix product `A^{-1} @ x`.
        """
        if isinstance(self.operator, LinearOperator):
            return self.operator._solve(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
        return super()._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _solve(
        self,
        rhs: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Solve `A^{-1} X = rhs` via `A @ rhs`.

        Args:
            rhs: Right-hand side `Tensor`.
            adjoint: If `True`, solve with `(A^{-1})^H`.
            adjoint_arg: If `True`, solve with `rhs^H`.

        Returns:
            Tensor: Solution tensor `A @ rhs`.
        """
        if isinstance(self.operator, LinearOperator):
            return self.operator._matmul(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
        return super()._solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)


def _kronecker_product_2d(a: Tensor, b: Tensor) -> Tensor:
    """Compute the Kronecker product `a (x) b` of two 2-D matrix `Tensor`s.

    Args:
        a: First 2-D matrix `Tensor` of shape `[M, N]`.
        b: Second 2-D matrix `Tensor` of shape `[P, Q]`.

    Returns:
        Tensor: Kronecker product `Tensor` of shape `[M * P, N * Q]`.
    """
    mat_a = _to_nested_floats(a)
    mat_b = _to_nested_floats(b)
    m = len(mat_a)
    n = len(mat_a[0]) if mat_a else 0
    p = len(mat_b)
    q = len(mat_b[0]) if mat_b else 0
    grid: list[list[float]] = [[0.0 for _ in range(n * q)] for _ in range(m * p)]
    for i in range(m):
        for j in range(n):
            a_val = mat_a[i][j]
            for r in range(p):
                for s in range(q):
                    grid[i * p + r][j * q + s] = a_val * mat_b[r][s]
    return Tensor(grid, dtype=a.dtype)


class LinearOperatorKronecker(LinearOperator):
    """`LinearOperator` acting like the Kronecker product of a sequence of operators."""

    def __init__(
        self,
        operators: Sequence[LinearOperator | Tensor | TensorOperand] = (((1.0,),),),
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize `LinearOperatorKronecker`.

        Args:
            operators: Sequence of `LinearOperator` or matrix tensors.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Optional name for the operator.
        """
        self.operators = list(operators)
        t_mat = self._compute_kronecker_dense()
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name or "LinearOperatorKronecker",
            matrix=t_mat,
        )

    def _compute_kronecker_dense(self) -> Tensor:
        """Compute the full Kronecker product across `self.operators`.

        Returns:
            Tensor: Dense Kronecker product matrix.
        """
        mats = [_extract_operator_dense(op) for op in self.operators]
        if not mats:
            return Tensor([[1.0]])
        acc = mats[0]
        for m in mats[1:]:
            acc = _kronecker_product_2d(acc, m)
        return acc

    def _to_dense(self) -> Tensor:
        """Return the Kronecker product dense matrix.

        Returns:
            Tensor: Dense Kronecker product matrix.
        """
        return self._compute_kronecker_dense()

    def _trace(self) -> Tensor:
        """Compute trace as the product of traces of each factor operator.

        Returns:
            Tensor: Trace tensor.
        """
        if not self.operators:
            return Tensor(1.0, dtype=self.dtype)
        traces = [
            op.trace()
            if isinstance(op, LinearOperator)
            else linalg.trace(_extract_operator_dense(op))
            for op in self.operators
        ]
        acc = _to_tensor(traces[0])
        for tr in traces[1:]:
            acc = _msc_ops.multiply(acc, _to_tensor(tr))
        return _wrap_tensor(acc)


class LinearOperatorLowRankUpdate(LinearOperator):
    """`LinearOperator` formed by perturbing `base_operator` with `U D V^H`."""

    def __init__(
        self,
        base_operator: LinearOperator | Tensor | TensorOperand = ((1.0,),),
        u: Tensor | TensorOperand = ((0.0,),),
        diag_update: Tensor | TensorOperand = None,
        v: Tensor | TensorOperand = None,
        is_diag_update_positive: bool | None = None,
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = True,
        is_positive_definite: bool | None = True,
        is_square: bool | None = None,
        name: str = "LinearOperatorLowRankUpdate",
    ) -> None:
        """Initialize `LinearOperatorLowRankUpdate`.

        Args:
            base_operator: Base `LinearOperator` or matrix `A`.
            u: Left low-rank update matrix `U`.
            diag_update: Optional diagonal update vector `D`.
            v: Optional right low-rank update matrix `V` (defaults to `u`).
            is_diag_update_positive: Hint that `diag_update` is positive.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        self.base_operator = base_operator
        self.u = Tensor(u)
        self.diag_update = diag_update
        self.v = Tensor(u if v is None else v)
        self.is_diag_update_positive = is_diag_update_positive
        t_mat = self._compute_low_rank_dense()
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _compute_low_rank_dense(self) -> Tensor:
        """Compute `A + U @ D @ V^H`.

        Returns:
            Tensor: Perturbed dense matrix tensor.
        """
        base_mat = _extract_operator_dense(self.base_operator)
        u_t = self.u
        v_t = self.v
        if self.diag_update is not None:
            d_mat = linalg.diag(Tensor(self.diag_update, dtype=base_mat.dtype))
            u_t = linalg.matmul(u_t, d_mat)
        pert = linalg.matmul(u_t, v_t, adjoint_b=True)
        return _wrap_tensor(_msc_ops.add(_to_tensor(base_mat), _to_tensor(pert)))

    def _to_dense(self) -> Tensor:
        """Return `A + U D V^H` as a dense `Tensor`.

        Returns:
            Tensor: Dense matrix tensor.
        """
        return self._compute_low_rank_dense()

    def _matmul(
        self,
        x: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Compute `(A + U D V^H) @ x` using low-rank associative multiplication.

        Args:
            x: Right-hand matrix `Tensor`.
            adjoint: If `True`, multiply by the adjoint.
            adjoint_arg: If `True`, multiply by the adjoint of `x`.

        Returns:
            Tensor: Matrix product tensor.
        """
        rhs = linalg.adjoint(x) if adjoint_arg else x
        base_part = (
            self.base_operator.matmul(rhs, adjoint=adjoint)
            if isinstance(self.base_operator, LinearOperator)
            else linalg.matmul(_extract_operator_dense(self.base_operator), rhs, adjoint_a=adjoint)
        )
        left_f = self.v if adjoint else self.u
        right_f = self.u if adjoint else self.v
        vx = linalg.matmul(right_f, rhs, adjoint_a=True)
        if self.diag_update is not None:
            d_col = _ops.expand_dims(_to_tensor(Tensor(self.diag_update, dtype=self.dtype)), -1)
            vx = _wrap_tensor(_msc_ops.multiply(d_col, _to_tensor(vx)))
        udvx = linalg.matmul(left_f, vx)
        return _wrap_tensor(_msc_ops.add(_to_tensor(base_part), _to_tensor(udvx)))


def _build_permutation_dense(perm: Tensor, dtype: DType) -> Tensor:
    """Build a row-permutation matrix `P` where `P[i, perm[i]] = 1`.

    Args:
        perm: Permutation index tensor.
        dtype: Target matrix `DType`.

    Returns:
        Tensor: Dense permutation matrix `Tensor`.
    """
    if not perm.shape:
        return Tensor([[1.0]], dtype=dtype)
    idx_list = [int(x) for x in _to_1d_floats(perm)]
    n = len(idx_list)
    grid: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, c in enumerate(idx_list):
        grid[i][c % n] = 1.0
    return Tensor(grid, dtype=dtype)


class LinearOperatorPermutation(LinearOperator):
    """`LinearOperator` acting like a row-permutation matrix."""

    def __init__(
        self,
        perm: Tensor | TensorOperand = (0,),
        dtype: DTypeLike = "float32",
        is_non_singular: bool | None = True,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = True,
        name: str = "LinearOperatorPermutation",
    ) -> None:
        """Initialize `LinearOperatorPermutation`.

        Args:
            perm: Permutation index vector of length `N`.
            dtype: Data type of the operator.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        self.perm = Tensor(perm, dtype="int32")
        resolved_dtype = to_dtype(dtype)
        t_mat = _build_permutation_dense(self.perm, resolved_dtype)
        super().__init__(
            dtype=resolved_dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return the row-permutation dense matrix.

        Returns:
            Tensor: Dense permutation matrix tensor.
        """
        return _build_permutation_dense(self.perm, self.dtype)

    def _matmul(
        self,
        x: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Permute rows of `x` according to `self.perm` (or inverse permutation if `adjoint`).

        Args:
            x: Right-hand matrix `Tensor`.
            adjoint: If `True`, apply the inverse permutation `P^T`.
            adjoint_arg: If `True`, multiply by the adjoint of `x`.

        Returns:
            Tensor: Permuted matrix tensor.
        """
        if not self.perm.shape:
            return super()._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
        rhs = linalg.adjoint(x) if adjoint_arg else x
        p_idx = _ops.argsort(_to_tensor(self.perm)) if adjoint else _to_tensor(self.perm)
        return _wrap_tensor(_ops.take(_to_tensor(rhs), p_idx, axis=-2))

    def _log_abs_determinant(self) -> Tensor:
        """Return `0.0` since permutation matrices have determinant `+-1`.

        Returns:
            Tensor: Scalar `0.0` tensor.
        """
        return Tensor(0.0, dtype=self.dtype)


class LinearOperatorScaledIdentity(LinearOperator):
    """`LinearOperator` acting like a scaled identity matrix `multiplier * I`."""

    def __init__(
        self,
        num_rows: int = 1,
        multiplier: Tensor | TensorOperand = 1.0,
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool = True,
        assert_proper_shapes: bool = False,
        name: str = "LinearOperatorScaledIdentity",
    ) -> None:
        """Initialize `LinearOperatorScaledIdentity`.

        Args:
            num_rows: Number of rows and columns in the identity matrix.
            multiplier: Scalar multiplier tensor.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            assert_proper_shapes: Whether to assert proper shapes.
            name: Name of the operator.
        """
        del assert_proper_shapes
        self._num_rows = int(num_rows)
        self.multiplier = Tensor(multiplier)
        dt_str = self.multiplier.dtype.value if self.multiplier.dtype is not None else "float32"
        t_mat = linalg.eye(self._num_rows, dtype=dt_str) * self.multiplier
        super().__init__(
            dtype=self.multiplier.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return `multiplier * I`.

        Returns:
            Tensor: Scaled identity dense matrix.
        """
        return linalg.eye(self._num_rows, dtype=self.dtype.value) * self.multiplier

    def _diag_part(self) -> Tensor:
        """Return vector of length `num_rows` filled with `multiplier`.

        Returns:
            Tensor: Diagonal vector tensor.
        """
        ones_v = _ops.ones((self._num_rows,), dtype=self.dtype.value)
        return _wrap_tensor(_msc_ops.multiply(ones_v, _to_tensor(self.multiplier)))

    def _trace(self) -> Tensor:
        """Return `num_rows * multiplier`.

        Returns:
            Tensor: Trace tensor.
        """
        return _wrap_tensor(
            _msc_ops.multiply(
                _to_tensor(float(self._num_rows), dtype=self.dtype),
                _to_tensor(self.multiplier),
            )
        )

    def _determinant(self) -> Tensor:
        """Return `multiplier ** num_rows`.

        Returns:
            Tensor: Determinant tensor.
        """
        return _wrap_tensor(
            _ops.power(
                _to_tensor(self.multiplier),
                _to_tensor(float(self._num_rows), dtype=self.dtype),
            )
        )

    def _matmul(
        self,
        x: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Multiply `x` by scalar `multiplier`.

        Args:
            x: Right-hand matrix `Tensor`.
            adjoint: Unused for real scalar multiplier.
            adjoint_arg: If `True`, multiply by the adjoint of `x`.

        Returns:
            Tensor: Scaled matrix tensor.
        """
        del adjoint
        rhs = linalg.adjoint(x) if adjoint_arg else x
        return _wrap_tensor(_msc_ops.multiply(_to_tensor(self.multiplier), _to_tensor(rhs)))

    def _solve(
        self,
        rhs: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Divide `rhs` by scalar `multiplier`.

        Args:
            rhs: Right-hand side `Tensor`.
            adjoint: Unused for real scalar multiplier.
            adjoint_arg: If `True`, solve with the adjoint of `rhs`.

        Returns:
            Tensor: Solution tensor.
        """
        del adjoint
        b = linalg.adjoint(rhs) if adjoint_arg else rhs
        return _wrap_tensor(_msc_ops.divide(_to_tensor(b), _to_tensor(self.multiplier)))


def _build_toeplitz_dense(col: Tensor, row: Tensor) -> Tensor:
    """Build a Toeplitz matrix where `T[i, j] = col[i - j]` for `i >= j` and `row[j - i]` for `i < j`.

    Args:
        col: First column vector `Tensor`.
        row: First row vector `Tensor`.

    Returns:
        Tensor: Dense Toeplitz matrix `Tensor`.
    """
    c_vals = _to_1d_floats(col)
    r_vals = _to_1d_floats(row)
    m = len(c_vals)
    n = len(r_vals)
    grid: list[list[float]] = [
        [c_vals[i - j] if i >= j else r_vals[j - i] for j in range(n)] for i in range(m)
    ]
    return Tensor(grid, dtype=col.dtype)


class LinearOperatorToeplitz(LinearOperator):
    """`LinearOperator` acting like a Toeplitz matrix defined by its first column and row."""

    def __init__(
        self,
        col: Tensor | TensorOperand = (1.0,),
        row: Tensor | TensorOperand = (1.0,),
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = None,
        name: str = "LinearOperatorToeplitz",
    ) -> None:
        """Initialize `LinearOperatorToeplitz`.

        Args:
            col: 1-D vector tensor specifying the first column.
            row: 1-D vector tensor specifying the first row.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        self.col = Tensor(col)
        self.row = Tensor(row)
        t_mat = _build_toeplitz_dense(self.col, self.row)
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return the Toeplitz dense matrix.

        Returns:
            Tensor: Dense Toeplitz matrix tensor.
        """
        return _build_toeplitz_dense(self.col, self.row)


class LinearOperatorTridiag(LinearOperator):
    """`LinearOperator` acting like a tridiagonal matrix."""

    def __init__(
        self,
        diagonals: Tensor | TensorOperand = ((0.0,), (1.0,), (0.0,)),
        diagonals_format: str = "compact",
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = None,
        name: str = "LinearOperatorTridiag",
    ) -> None:
        """Initialize `LinearOperatorTridiag`.

        Args:
            diagonals: Tridiagonal specification (`compact`, `sequence`, or `matrix`).
            diagonals_format: Format of `diagonals`.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            name: Name of the operator.
        """
        self.diagonals = diagonals
        self.diagonals_format = diagonals_format
        t_mat = linalg._build_tridiagonal_matrix(diagonals, diagonals_format=diagonals_format)
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return the tridiagonal dense matrix.

        Returns:
            Tensor: Dense tridiagonal matrix tensor.
        """
        return linalg._build_tridiagonal_matrix(
            self.diagonals,
            diagonals_format=self.diagonals_format,
        )


class LinearOperatorZeros(LinearOperator):
    """`LinearOperator` acting like a zero matrix of shape `[..., num_rows, num_columns]`."""

    def __init__(
        self,
        num_rows: int = 1,
        num_columns: int | None = None,
        batch_shape: IntTuple | None = None,
        dtype: DTypeLike = None,
        is_non_singular: bool = False,
        is_self_adjoint: bool = True,
        is_positive_definite: bool = False,
        is_square: bool = True,
        assert_proper_shapes: bool = False,
        name: str = "LinearOperatorZeros",
    ) -> None:
        """Initialize `LinearOperatorZeros`.

        Args:
            num_rows: Number of rows.
            num_columns: Optional number of columns (defaults to `num_rows`).
            batch_shape: Optional leading batch dimensions.
            dtype: Data type of the zero matrix.
            is_non_singular: Hint that operator is non-singular.
            is_self_adjoint: Hint that operator is self-adjoint.
            is_positive_definite: Hint that operator is positive-definite.
            is_square: Hint that operator is square.
            assert_proper_shapes: Whether to assert proper shapes.
            name: Name of the operator.
        """
        del assert_proper_shapes
        self._num_rows = int(num_rows)
        self._num_cols = self._num_rows if num_columns is None else int(num_columns)
        self._batch_shape: IntTuple = tuple(batch_shape) if batch_shape is not None else ()
        resolved_dtype = to_dtype(dtype)
        t_mat = _wrap_tensor(
            _ops.zeros(
                (*self._batch_shape, self._num_rows, self._num_cols),
                dtype=resolved_dtype.value,
            )
        )
        super().__init__(
            dtype=resolved_dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )

    def _to_dense(self) -> Tensor:
        """Return the zero matrix tensor.

        Returns:
            Tensor: Zero matrix tensor.
        """
        return self._matrix

    def _determinant(self) -> Tensor:
        """Return `0.0` as the determinant of a zero matrix.

        Returns:
            Tensor: Zero determinant tensor.
        """
        return _wrap_tensor(_ops.zeros(self._batch_shape, dtype=self.dtype.value))

    def _trace(self) -> Tensor:
        """Return `0.0` as the trace of a zero matrix.

        Returns:
            Tensor: Zero trace tensor.
        """
        return _wrap_tensor(_ops.zeros(self._batch_shape, dtype=self.dtype.value))

    def _matmul(
        self,
        x: Tensor,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ) -> Tensor:
        """Multiply by zero matrix, returning a zero tensor of appropriate shape.

        Args:
            x: Right-hand matrix `Tensor`.
            adjoint: If `True`, transpose dimensions of zero matrix.
            adjoint_arg: If `True`, transpose `x`.

        Returns:
            Tensor: Zero product tensor.
        """
        rhs = linalg.adjoint(x) if adjoint_arg else x
        out_rows = self._num_cols if adjoint else self._num_rows
        out_cols = rhs.shape[-1] if rhs.shape else 1
        return _wrap_tensor(
            _ops.zeros((*self._batch_shape, out_rows, out_cols), dtype=self.dtype.value)
        )


def _lu_single_2d(
    mat_t: Tensor,
    idx_dtype: str,
) -> builtins.tuple[Tensor, Tensor]:
    """Compute partial-pivoting PLU decomposition `P @ A = L @ U` (or `A = P^{-1} L U`) for a 2-D matrix.

    Args:
        mat_t: 2-D square matrix `Tensor`.
        idx_dtype: Integer dtype string for permutation vector `p`.

    Returns:
        tuple[Tensor, Tensor]: `(lu, p)` tensors for the 2-D matrix.
    """
    mat = _to_nested_floats(mat_t)
    n = len(mat)
    perm = list(range(n))
    for k in range(n):
        # Partial pivoting: find row with largest magnitude in column k
        piv_row = k
        max_abs = abs(mat[k][k])
        for r in range(k + 1, n):
            if abs(mat[r][k]) > max_abs:
                max_abs = abs(mat[r][k])
                piv_row = r
        if piv_row != k:
            mat[k], mat[piv_row] = mat[piv_row], mat[k]
            perm[k], perm[piv_row] = perm[piv_row], perm[k]
        piv = mat[k][k] if abs(mat[k][k]) > 1e-12 else 1.0
        for i in range(k + 1, n):
            factor = mat[i][k] / piv
            mat[i][k] = factor
            for j in range(k + 1, n):
                mat[i][j] -= factor * mat[k][j]
    # TensorFlow's tf.linalg.lu returns `p` such that P @ L @ U = A, where P[i, p[i]] = 1,
    # i.e. p is the inverse permutation of `perm`.
    inv_perm = [0] * n
    for i, p_val in enumerate(perm):
        inv_perm[p_val] = i
    return (Tensor(mat, dtype=mat_t.dtype), Tensor(inv_perm, dtype=idx_dtype))


class linalg:
    """TensorFlow `linalg` namespace."""

    LinearOperator = LinearOperator
    LinearOperatorAdjoint = LinearOperatorAdjoint
    LinearOperatorBlockDiag = LinearOperatorBlockDiag
    LinearOperatorBlockLowerTriangular = LinearOperatorBlockLowerTriangular
    LinearOperatorCirculant = LinearOperatorCirculant
    LinearOperatorCirculant2D = LinearOperatorCirculant2D
    LinearOperatorCirculant3D = LinearOperatorCirculant3D
    LinearOperatorComposition = LinearOperatorComposition
    LinearOperatorDiag = LinearOperatorDiag
    LinearOperatorFullMatrix = LinearOperatorFullMatrix
    LinearOperatorHouseholder = LinearOperatorHouseholder
    LinearOperatorIdentity = LinearOperatorIdentity
    LinearOperatorInversion = LinearOperatorInversion
    LinearOperatorKronecker = LinearOperatorKronecker
    LinearOperatorLowRankUpdate = LinearOperatorLowRankUpdate
    LinearOperatorLowerTriangular = LinearOperatorLowerTriangular
    LinearOperatorPermutation = LinearOperatorPermutation
    LinearOperatorScaledIdentity = LinearOperatorScaledIdentity
    LinearOperatorToeplitz = LinearOperatorToeplitz
    LinearOperatorTridiag = LinearOperatorTridiag
    LinearOperatorZeros = LinearOperatorZeros

    @staticmethod
    def adjoint(
        matrix: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Transpose and conjugate the innermost two dimensions of `matrix`.

        Args:
            matrix: Input matrix tensor of rank `>= 2`.
            name: Optional operation name.

        Returns:
            Tensor: Adjoint (conjugate transpose) matrix tensor.
        """
        del name
        return _wrap_tensor(as_operand(_msc_ops.adjoint(_to_tensor(matrix))))

    @staticmethod
    def band_part(
        input: Tensor | TensorOperand,
        num_lower: int,
        num_upper: int,
        name: str | None = None,
    ) -> Tensor:
        """Copy a tensor setting everything outside a central diagonal band to zero.

        Args:
            input: Input matrix tensor of shape `[..., M, N]`.
            num_lower: Number of subdiagonals to keep (negative keeps all).
            num_upper: Number of superdiagonals to keep (negative keeps all).
            name: Optional operation name.

        Returns:
            Tensor: Banded matrix tensor.
        """
        del name
        t_in = _to_tensor(input)
        if len(t_in.shape) < 2:
            return _wrap_tensor(t_in)
        rows = t_in.shape[-2]
        cols = t_in.shape[-1]
        r_idx = _ops.expand_dims(_ops.arange(rows), 1)
        c_idx = _ops.expand_dims(_ops.arange(cols), 0)
        cond_lower = (
            _ops.ones((rows, cols), dtype="bool")
            if num_lower < 0
            else _msc_ops.less_equal(_msc_ops.subtract(r_idx, c_idx), _to_tensor(num_lower))
        )
        cond_upper = (
            _ops.ones((rows, cols), dtype="bool")
            if num_upper < 0
            else _msc_ops.less_equal(_msc_ops.subtract(c_idx, r_idx), _to_tensor(num_upper))
        )
        mask = _msc_ops.logical_and(cond_lower, cond_upper)
        return _wrap_tensor(_ops.where(mask, t_in, _ops.zeros_like(t_in)))

    @staticmethod
    def banded_triangular_solve(
        bands: Tensor | TensorOperand,
        rhs: Tensor | TensorOperand,
        lower: bool = True,
        adjoint: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Solve triangular systems of equations with a banded solver.

        Args:
            bands: Banded triangular matrix tensor.
            rhs: Right-hand-side tensor.
            lower: Whether `bands` represents a lower-triangular matrix.
            adjoint: Whether to solve with the conjugate transpose.
            name: Optional operation name.

        Returns:
            Tensor: Solution tensor.
        """
        return linalg.triangular_solve(bands, rhs, lower=lower, adjoint=adjoint, name=name)

    @staticmethod
    def cholesky(
        input: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the Cholesky decomposition of one or more square matrices.

        Args:
            input: Hermitian positive-definite matrix tensor of shape `[..., M, M]`.
            name: Optional operation name.

        Returns:
            Tensor: Lower-triangular Cholesky factor tensor.
        """
        del name
        t_in = Tensor(input)
        if _tracer.is_tracing or len(t_in.shape) <= 2:
            return _wrap_tensor(_ops.cholesky(_to_tensor(t_in)))
        batch_shape = t_in.shape[:-2]
        inner_shape = t_in.shape[-2:]
        slices = [
            _wrap_tensor(_ops.cholesky(_to_tensor(_slice_batch_matrix(t_in, idx))))
            for idx in _iter_batch_indices(batch_shape)
        ]
        return _stack_batch_results(slices, batch_shape, inner_shape, t_in.dtype)

    @staticmethod
    def cholesky_solve(
        chol: Tensor | TensorOperand,
        rhs: Tensor | TensorOperand,
        name: str = "cholesky_solve",
    ) -> Tensor:
        """Solve systems of linear equations `A X = rhs` given Cholesky factor `chol` of `A`.

        Args:
            chol: Lower-triangular Cholesky factor `L` such that `A = L @ L^H`.
            rhs: Right-hand-side tensor.
            name: Optional operation name.

        Returns:
            Tensor: Solution tensor `X`.
        """
        del name
        y = linalg.triangular_solve(chol, rhs, lower=True, adjoint=False)
        return linalg.triangular_solve(chol, y, lower=True, adjoint=True)

    @staticmethod
    def cross(
        a: Tensor | TensorOperand,
        b: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the pairwise 3-D cross product of `a` and `b` across arbitrary batch dimensions.

        Args:
            a: First 3-element vector tensor of shape `[..., 3]`.
            b: Second 3-element vector tensor of shape `[..., 3]`.
            name: Optional operation name.

        Returns:
            Tensor: Cross product tensor of shape `[..., 3]`.
        """
        del name
        return _wrap_tensor(_ops.cross(_to_tensor(a), _to_tensor(b)))

    @staticmethod
    def det(
        input: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the determinant of one or more square matrices.

        Args:
            input: Square matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor: Determinant tensor.
        """
        del name
        return _wrap_tensor(_ops.det(_to_tensor(input)))

    @staticmethod
    def diag(
        diagonal: Tensor | TensorOperand,
        name: str = "diag",
        k: int | builtins.tuple[int, int] = 0,
        num_rows: int = -1,
        num_cols: int = -1,
        padding_value: float = 0,
        align: str = "RIGHT_LEFT",
    ) -> Tensor:
        """Return a batched diagonal tensor with given batched diagonal values.

        Args:
            diagonal: Rank `r` tensor of diagonal values.
            name: Optional operation name.
            k: Diagonal offset (`0` for main diagonal, positive for superdiagonal, negative for subdiagonal).
            num_rows: Number of rows of the output matrix (`-1` infers automatically).
            num_cols: Number of columns of the output matrix (`-1` infers automatically).
            padding_value: Value to fill the area outside the specified diagonal band.
            align: Alignment string for sub-/super-diagonals.

        Returns:
            Tensor: Diagonal matrix tensor of shape `[..., M, N]`.
        """
        del name, align
        k_val = k[0] if isinstance(k, builtins.tuple) else int(k)
        t_diag = Tensor(diagonal)
        if _tracer.is_tracing and len(t_diag.shape) <= 1:
            d_len = t_diag.shape[-1] if t_diag.shape else 1
            raw_d = _to_tensor(t_diag)
            eye_m = _msc_ops.eye(d_len, dtype=raw_d.dtype)
            main_d = _msc_ops.multiply(eye_m, _ops.expand_dims(raw_d, -1))
            if k_val == 0:
                return _wrap_tensor(main_d)
            return linalg.band_part(
                _ops.roll(main_d, shift=k_val, axis=-1),
                max(0, -k_val),
                max(0, k_val),
            )
        if len(t_diag.shape) <= 1 and num_rows == -1 and num_cols == -1 and padding_value == 0:
            if k_val == 0:
                return _wrap_tensor(_ops.diag(_to_tensor(t_diag)))
            return _wrap_tensor(_ops.diag(_to_tensor(t_diag), k_val))

        batch_shape = t_diag.shape[:-1]
        diag_len = t_diag.shape[-1] if t_diag.shape else 1
        inferred_size = diag_len + abs(k_val)
        m_rows = inferred_size if num_rows < 0 else num_rows
        n_cols = inferred_size if num_cols < 0 else num_cols

        slices: list[Tensor] = []
        for idx in _iter_batch_indices(batch_shape):
            cur = t_diag
            for b_i in idx:
                cur = cur[b_i]
            vals = _to_1d_floats(cur)
            grid: list[list[float]] = [
                [float(padding_value) for _ in range(n_cols)] for _ in range(m_rows)
            ]
            r_start = max(0, -k_val)
            c_start = max(0, k_val)
            for d_i, v in enumerate(vals):
                grid[r_start + d_i][c_start + d_i] = v
            slices.append(Tensor(grid, dtype=t_diag.dtype))
        if not batch_shape:
            return slices[0]
        return _stack_batch_results(slices, batch_shape, (m_rows, n_cols), t_diag.dtype)

    @staticmethod
    def diag_part(
        input: Tensor | TensorOperand,
        name: str | None = None,
        k: int | builtins.tuple[int, int] = 0,
        padding_value: float = 0.0,
        align: str = "RIGHT_LEFT",
    ) -> Tensor:
        """Extract the diagonal part of a batched matrix tensor.

        Args:
            input: Input matrix tensor of shape `[..., M, N]`.
            name: Optional operation name.
            k: Diagonal offset (`int` or `(k_low, k_high)` tuple).
            padding_value: Padding value for rectangular diagonals.
            align: Alignment string.

        Returns:
            Tensor: Extracted diagonal tensor.
        """
        del name, align
        t_in = Tensor(input)
        if isinstance(k, int) and (_tracer.is_tracing or len(t_in.shape) <= 2):
            return _wrap_tensor(as_operand(_msc_ops.tensor_diag_part(_to_tensor(t_in), offset=k)))

        batch_shape = t_in.shape[:-2]
        m_rows = t_in.shape[-2]
        n_cols = t_in.shape[-1]

        if isinstance(k, builtins.tuple):
            k_low, k_high = int(k[0]), int(k[1])
        else:
            k_low = k_high = int(k)

        max_diag_len = min(m_rows, n_cols)
        slices: list[Tensor] = []
        for idx in _iter_batch_indices(batch_shape):
            mat_2d = _to_nested_floats(_slice_batch_matrix(t_in, idx))
            band_diags: list[list[float]] = []
            for offset in range(k_high, k_low - 1, -1):
                r_start = max(0, -offset)
                c_start = max(0, offset)
                d_len = max(0, min(m_rows - r_start, n_cols - c_start))
                diag_vals = [mat_2d[r_start + d_i][c_start + d_i] for d_i in range(d_len)]
                if k_low != k_high and len(diag_vals) < max_diag_len:
                    diag_vals.extend([float(padding_value)] * (max_diag_len - len(diag_vals)))
                band_diags.append(diag_vals)
            slices.append(
                Tensor(band_diags[0] if k_low == k_high else band_diags, dtype=t_in.dtype)
            )

        if not batch_shape:
            return slices[0]
        return _stack_batch_results(slices, batch_shape, slices[0].shape, t_in.dtype)

    @staticmethod
    def eig(
        tensor: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
        """Compute the eigendecomposition of a batch of matrices.

        Args:
            tensor: Square matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor | tuple[Tensor, ...] | list[Tensor]: Pair `(eigenvalues, eigenvectors)`.
        """
        del name
        return linalg.eigh(tensor)

    @staticmethod
    def eigvals(
        tensor: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the eigenvalues of one or more matrices.

        Args:
            tensor: Square matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor: Eigenvalues tensor.
        """
        return linalg.eigvalsh(tensor, name=name)

    @staticmethod
    def eigh(
        tensor: Tensor | TensorOperand,
        name: str | None = None,
    ) -> builtins.tuple[Tensor, Tensor]:
        """Compute the eigendecomposition of a batch of self-adjoint matrices.

        Args:
            tensor: Self-adjoint matrix tensor of shape `[..., N, N]`.
            name: Optional operation name.

        Returns:
            tuple[Tensor, Tensor]: Pair `(eigenvalues, eigenvectors)`.
        """
        del name
        t_in = Tensor(tensor)
        if _tracer.is_tracing or len(t_in.shape) <= 2:
            raw_res = _ops.eigh(_to_tensor(t_in))
            return (_wrap_tensor(raw_res[0]), _wrap_tensor(raw_res[1]))
        batch_shape = t_in.shape[:-2]
        n = t_in.shape[-1]
        val_slices: list[Tensor] = []
        vec_slices: list[Tensor] = []
        for idx in _iter_batch_indices(batch_shape):
            sub_res = _ops.eigh(_to_tensor(_slice_batch_matrix(t_in, idx)))
            val_slices.append(_wrap_tensor(sub_res[0]))
            vec_slices.append(_wrap_tensor(sub_res[1]))
        return (
            _stack_batch_results(val_slices, batch_shape, (n,), t_in.dtype),
            _stack_batch_results(vec_slices, batch_shape, (n, n), t_in.dtype),
        )

    @staticmethod
    def eigh_tridiagonal(
        alpha: Tensor | TensorOperand,
        beta: Tensor | TensorOperand,
        eigvals_only: bool = True,
        select: str = "a",
        select_range: builtins.tuple[float, float] | None = None,
        tol: float | None = None,
        name: str | None = None,
    ) -> Tensor | builtins.tuple[Tensor, Tensor]:
        """Compute the eigenvalues (and optionally eigenvectors) of a symmetric tridiagonal matrix.

        Args:
            alpha: 1-D tensor of main diagonal elements of length `N`.
            beta: 1-D tensor of off-diagonal elements of length `N - 1`.
            eigvals_only: If `True`, only compute eigenvalues.
            select: Selection mode (`'a'`, `'v'`, or `'i'`).
            select_range: Range of eigenvalues or indices when `select != 'a'`.
            tol: Optional absolute tolerance.
            name: Optional operation name.

        Returns:
            Tensor | tuple[Tensor, Tensor]: Eigenvalues or `(eigenvalues, eigenvectors)`.
        """
        del select, select_range, tol, name
        t_alpha = Tensor(alpha)
        t_beta = Tensor(beta)
        n = t_alpha.shape[-1] if t_alpha.shape else 1
        main_m = _to_tensor(linalg.diag(t_alpha))
        if n > 1 and t_beta.shape and t_beta.shape[-1] > 0:
            off_m = _to_tensor(linalg.diag(t_beta, k=1))
            off_m_t = _to_tensor(linalg.diag(t_beta, k=-1))
            mat = _wrap_tensor(_msc_ops.add(_msc_ops.add(main_m, off_m), off_m_t))
        else:
            mat = _wrap_tensor(main_m)
        if eigvals_only:
            return linalg.eigvalsh(mat)
        return linalg.eigh(mat)

    @staticmethod
    def eigvalsh(
        tensor: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the eigenvalues of one or more self-adjoint matrices.

        Args:
            tensor: Self-adjoint matrix tensor of shape `[..., N, N]`.
            name: Optional operation name.

        Returns:
            Tensor: Eigenvalues tensor of shape `[..., N]`.
        """
        del name
        vals, _ = linalg.eigh(tensor)
        return vals

    @staticmethod
    def einsum(
        equation: str,
        *inputs: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Tensor contraction over specified indices and outer product (`tf.linalg.einsum`).

        Args:
            equation: Einstein summation notation string.
            *inputs: Input tensors to contract.
            **kwargs: Optional keyword arguments (such as `optimize` or `name`).

        Returns:
            Tensor: Contracted result tensor.
        """
        del kwargs
        res = _ops.einsum(equation, *[_to_tensor(a) for a in inputs])
        return _wrap_tensor(res)

    @staticmethod
    def expm(
        input: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the matrix exponential of one or more square matrices via scaling and squaring.

        Args:
            input: Square matrix tensor of shape `[..., M, M]`.
            name: Optional operation name.

        Returns:
            Tensor: Matrix exponential tensor of shape `[..., M, M]`.
        """
        del name
        t_in = Tensor(input)

        def _expm_2d(mat_2d: Tensor) -> Tensor:
            """Compute the matrix exponential of a 2-D matrix via scaling-and-squaring Horner Taylor series.

            Args:
                mat_2d: 2-D square matrix `Tensor`.

            Returns:
                Tensor: 2-D matrix exponential `Tensor`.
            """
            t_a = _to_tensor(mat_2d)
            m = mat_2d.shape[-1]
            eye_m = _ops.eye(m, m, dtype=t_a.dtype)
            scaled = _ops.multiply(t_a, _to_tensor(1.0 / 16.0, dtype=t_a.dtype))
            acc = eye_m
            for k in range(12, 0, -1):
                step = _ops.multiply(
                    _ops.matmul(scaled, acc),
                    _to_tensor(1.0 / float(k), dtype=t_a.dtype),
                )
                acc = _ops.add(eye_m, step)
            for _ in range(4):
                acc = _ops.matmul(acc, acc)
            return _wrap_tensor(acc)

        if _tracer.is_tracing or len(t_in.shape) <= 2:
            return _expm_2d(t_in)
        batch_shape = t_in.shape[:-2]
        inner_shape = t_in.shape[-2:]
        slices = [
            _expm_2d(_slice_batch_matrix(t_in, idx)) for idx in _iter_batch_indices(batch_shape)
        ]
        return _stack_batch_results(slices, batch_shape, inner_shape, t_in.dtype)

    @staticmethod
    def eye(
        num_rows: int,
        num_columns: int | None = None,
        batch_shape: IntTuple | None = None,
        dtype: DTypeLike = "float32",
        name: str = "eye",
    ) -> Tensor:
        """Construct an identity matrix, or a batch of matrices.

        Args:
            num_rows: Non-negative integer number of rows.
            num_columns: Optional number of columns (defaults to `num_rows`).
            batch_shape: Optional leading batch dimensions.
            dtype: Data type of the resulting matrix.
            name: Optional operation name.

        Returns:
            Tensor: Identity matrix tensor of shape `[*batch_shape, num_rows, num_columns]`.
        """
        del name
        resolved_dtype = to_dtype(dtype)
        cols = num_rows if num_columns is None else num_columns
        base_eye = (
            _wrap_tensor(_ops.eye(num_rows, dtype=resolved_dtype.value))
            if num_columns is None
            else _wrap_tensor(_ops.eye(num_rows, cols, dtype=resolved_dtype.value))
        )
        if not batch_shape:
            return base_eye
        b_tuple = tuple(int(d) for d in batch_shape)
        b_count = 1
        for d in b_tuple:
            b_count *= d
        return _stack_batch_results(
            [base_eye for _ in range(b_count)],
            b_tuple,
            (num_rows, cols),
            resolved_dtype,
        )

    @staticmethod
    def global_norm(
        t_list: builtins.list[Tensor | TensorOperand] | builtins.tuple[Tensor | TensorOperand, ...],
        name: str | None = None,
    ) -> Tensor:
        """Compute the global Euclidean norm of multiple tensors.

        Args:
            t_list: Sequence of tensors.
            name: Optional operation name.

        Returns:
            Tensor: Scalar global L2 norm tensor.
        """
        del name
        items = [_to_tensor(t) for t in t_list if t is not None]
        if not items:
            return Tensor(0.0)
        sq_sums = [_ops.sum(_msc_ops.multiply(t, t)) for t in items]
        total_sq = sq_sums[0]
        for s in sq_sums[1:]:
            total_sq = _msc_ops.add(total_sq, s)
        return _wrap_tensor(_ops.sqrt(total_sq))

    @staticmethod
    def inv(
        input: Tensor | TensorOperand,
        adjoint: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Compute the inverse of one or more square invertible matrices.

        Args:
            input: Square invertible matrix tensor of shape `[..., M, M]`.
            adjoint: If `True`, compute the inverse of the conjugate transpose.
            name: Optional operation name.

        Returns:
            Tensor: Inverse matrix tensor.
        """
        del name
        t_in = linalg.adjoint(input) if adjoint else Tensor(input)
        if _tracer.is_tracing or len(t_in.shape) <= 2:
            return _wrap_tensor(_ops.inv(_to_tensor(t_in)))
        batch_shape = t_in.shape[:-2]
        inner_shape = t_in.shape[-2:]
        slices = [
            _wrap_tensor(_ops.inv(_to_tensor(_slice_batch_matrix(t_in, idx))))
            for idx in _iter_batch_indices(batch_shape)
        ]
        return _stack_batch_results(slices, batch_shape, inner_shape, t_in.dtype)

    @staticmethod
    def l2_normalize(
        x: Tensor | TensorOperand,
        axis: int | Sequence[int] | None = None,
        epsilon: float = 1e-12,
        name: str | None = None,
        dim: int | None = None,
    ) -> Tensor:
        """Normalize along dimension `axis` using an L2 norm with `epsilon` lower bound.

        Args:
            x: Input tensor.
            axis: Dimension(s) along which to normalize.
            epsilon: Lower bound value for the squared norm.
            name: Optional operation name.
            dim: Deprecated alias for `axis`.

        Returns:
            Tensor: L2-normalized tensor.
        """
        del name
        eff_axis = dim if axis is None else axis
        t_x = _to_tensor(x)
        axis_val: int | builtins.tuple[int, ...] | None = (
            eff_axis
            if isinstance(eff_axis, int)
            else (
                builtins.tuple(int(a) for a in eff_axis)
                if isinstance(eff_axis, (builtins.list, builtins.tuple))
                else None
            )
        )
        sq_sum = _to_tensor(_ops.sum(_ops.square(t_x), axis=axis_val, keepdims=True))
        inv_norm = _ops.rsqrt(_ops.maximum(sq_sum, _to_tensor(epsilon, dtype=t_x.dtype)))
        return _wrap_tensor(_msc_ops.multiply(t_x, inv_norm))

    @staticmethod
    def logdet(
        matrix: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute log of the determinant of a Hermitian positive definite matrix.

        Args:
            matrix: Hermitian positive definite matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor: Log-determinant tensor.
        """
        del name
        return _wrap_tensor(_msc_ops.log(_ops.det(_to_tensor(matrix))))

    @staticmethod
    def logm(
        input: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the matrix logarithm of one or more square matrices.

        Args:
            input: Square matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor: Matrix logarithm tensor.
        """
        del name
        return _wrap_tensor(as_operand(_msc_ops.logm(_to_tensor(input))))

    @staticmethod
    def lstsq(
        matrix: Tensor | TensorOperand,
        rhs: Tensor | TensorOperand,
        l2_regularizer: float = 0.0,
        fast: bool = True,
        name: str | None = None,
    ) -> Tensor:
        """Solve one or more linear least-squares problems `(A^T A + l2 * I) X = A^T B`.

        Args:
            matrix: Coefficient matrix tensor of shape `[..., M, N]`.
            rhs: Right-hand-side tensor of shape `[..., M, K]`.
            l2_regularizer: L2 regularization scalar weight.
            fast: Whether to use the fast normal-equations / Cholesky solver.
            name: Optional operation name.

        Returns:
            Tensor: Least-squares solution tensor of shape `[..., N, K]`.
        """
        del fast, name
        t_mat = Tensor(matrix)
        t_rhs = Tensor(rhs)
        if _tracer.is_tracing:
            return _wrap_tensor(as_operand(_msc_ops.lstsq(_to_tensor(t_mat), _to_tensor(t_rhs))))
        if l2_regularizer > 0.0:
            n = t_mat.shape[-1]
            batch_s = t_mat.shape[:-2] if len(t_mat.shape) > 2 else None
            reg_eye = linalg.eye(n, batch_shape=batch_s, dtype=t_mat.dtype) * float(l2_regularizer)
            ata = linalg.matmul(t_mat, t_mat, adjoint_a=True) + reg_eye
            atb = linalg.matmul(t_mat, t_rhs, adjoint_a=True)
            return linalg.solve(ata, atb)
        ata_unreg = linalg.matmul(t_mat, t_mat, adjoint_a=True)
        atb_unreg = linalg.matmul(t_mat, t_rhs, adjoint_a=True)
        return linalg.solve(ata_unreg, atb_unreg)

    @staticmethod
    def lu(
        input: Tensor | TensorOperand,
        output_idx_type: DTypeLike = None,
        name: str | None = None,
    ) -> builtins.tuple[Tensor, Tensor]:
        """Compute the LU decomposition of one or more square matrices.

        Args:
            input: Square matrix tensor of shape `[..., N, N]`.
            output_idx_type: Optional integer `DType` for permutation indices (`int32` or `int64`).
            name: Optional operation name.

        Returns:
            tuple[Tensor, Tensor]: Pair `(lu, p)` of combined LU factors and permutation indices.
        """
        del name
        t_in = Tensor(input)
        idx_dtype = to_dtype(output_idx_type) if output_idx_type is not None else DType.Int32
        if _tracer.is_tracing or len(t_in.shape) < 2:
            if _tracer.is_tracing:
                _ = _msc_ops.lu(_to_tensor(t_in))
            return (t_in, Tensor([0], dtype=idx_dtype))

        if len(t_in.shape) == 2:
            return _lu_single_2d(t_in, idx_dtype.value)

        batch_shape = t_in.shape[:-2]
        n = t_in.shape[-1]
        lu_slices: list[Tensor] = []
        p_slices: list[Tensor] = []
        for idx in _iter_batch_indices(batch_shape):
            lu_2d, p_1d = _lu_single_2d(_slice_batch_matrix(t_in, idx), idx_dtype.value)
            lu_slices.append(lu_2d)
            p_slices.append(p_1d)
        return (
            _stack_batch_results(lu_slices, batch_shape, (n, n), t_in.dtype),
            _stack_batch_results(p_slices, batch_shape, (n,), idx_dtype),
        )

    @staticmethod
    def lu_reconstruct(
        lower_upper: Tensor | TensorOperand,
        perm: Tensor | TensorOperand,
        validate_args: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Reconstruct the original matrix from `lu` and `perm` factors.

        Args:
            lower_upper: Combined lower/upper triangular factor matrix from `linalg.lu`.
            perm: Permutation index vector from `linalg.lu`.
            validate_args: Whether to validate input arguments.
            name: Optional operation name.

        Returns:
            Tensor: Reconstructed matrix `P @ L @ U`.
        """
        del validate_args, name
        t_lu = _to_tensor(lower_upper)
        n = t_lu.shape[-1] if t_lu.shape else 1
        diag_vec = as_operand(_msc_ops.tensor_diag_part(t_lu))
        lower_strict = _msc_ops.subtract(
            _to_tensor(linalg.band_part(t_lu, -1, 0)),
            _ops.diag(_to_tensor(diag_vec)),
        )
        l_mat = _msc_ops.add(
            _to_tensor(as_operand(lower_strict)),
            _ops.eye(n, dtype=t_lu.dtype.value),
        )
        u_mat = _to_tensor(linalg.band_part(t_lu, 0, -1))
        prod = _ops.matmul(_to_tensor(l_mat), u_mat)
        return _wrap_tensor(_ops.take(prod, _to_tensor(perm), axis=0))

    @staticmethod
    def lu_matrix_inverse(
        lower_upper: Tensor | TensorOperand,
        perm: Tensor | TensorOperand,
        validate_args: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Compute the matrix inverse given LU decomposition factors `lower_upper` and `perm`.

        Args:
            lower_upper: Combined lower/upper triangular factor matrix from `linalg.lu`.
            perm: Permutation index vector from `linalg.lu`.
            validate_args: Whether to validate input arguments.
            name: Optional operation name.

        Returns:
            Tensor: Inverse matrix tensor.
        """
        mat = linalg.lu_reconstruct(lower_upper, perm, validate_args=validate_args, name=name)
        return linalg.inv(mat)

    @staticmethod
    def lu_solve(
        lower_upper: Tensor | TensorOperand,
        perm: Tensor | TensorOperand,
        rhs: Tensor | TensorOperand,
        validate_args: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Solve linear system `A @ X = rhs` given LU decomposition factors of `A`.

        Args:
            lower_upper: Combined lower/upper triangular factor matrix from `linalg.lu`.
            perm: Permutation index vector from `linalg.lu`.
            rhs: Right-hand side matrix or vector tensor.
            validate_args: Whether to validate input arguments.
            name: Optional operation name.

        Returns:
            Tensor: Solution tensor `X`.
        """
        mat = linalg.lu_reconstruct(lower_upper, perm, validate_args=validate_args, name=name)
        return linalg.solve(mat, rhs)

    @staticmethod
    def matmul(
        a: Tensor | TensorOperand,
        b: Tensor | TensorOperand,
        transpose_a: bool = False,
        transpose_b: bool = False,
        adjoint_a: bool = False,
        adjoint_b: bool = False,
        a_is_sparse: bool = False,
        b_is_sparse: bool = False,
        output_type: DTypeLike = None,
        name: str | None = None,
    ) -> Tensor:
        """Multiply matrix `a` by matrix `b`, producing `a @ b`.

        Args:
            a: Left-hand matrix tensor.
            b: Right-hand matrix tensor.
            transpose_a: If `True`, `a` is transposed before multiplication.
            transpose_b: If `True`, `b` is transposed before multiplication.
            adjoint_a: If `True`, `a` is conjugated and transposed before multiplication.
            adjoint_b: If `True`, `b` is conjugated and transposed before multiplication.
            a_is_sparse: Hint that `a` is sparse.
            b_is_sparse: Hint that `b` is sparse.
            output_type: Optional output `DType`.
            name: Optional operation name.

        Returns:
            Tensor: Matrix product tensor.
        """
        del a_is_sparse, b_is_sparse, name
        ta = _to_tensor(a)
        tb = _to_tensor(b)
        if adjoint_a:
            ta = _to_tensor(as_operand(_msc_ops.adjoint(ta)))
        elif transpose_a:
            ta = _to_tensor(as_operand(_msc_ops.matrix_transpose(ta)))
        if adjoint_b:
            tb = _to_tensor(as_operand(_msc_ops.adjoint(tb)))
        elif transpose_b:
            tb = _to_tensor(as_operand(_msc_ops.matrix_transpose(tb)))
        res = _ops.matmul(ta, tb)
        if output_type is not None:
            res = _ops.cast(res, to_dtype(output_type).value)
        return _wrap_tensor(res)

    @staticmethod
    def matrix_rank(
        a: Tensor | TensorOperand,
        tol: float | None = None,
        validate_args: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Compute the matrix rank of one or more matrices.

        Args:
            a: Matrix tensor of shape `[..., M, N]`.
            tol: Optional singular value threshold.
            validate_args: Whether to validate arguments.
            name: Optional operation name.

        Returns:
            Tensor: Integer matrix rank tensor.
        """
        del validate_args, name
        return _wrap_tensor(as_operand(_msc_ops.matrix_rank(_to_tensor(a), tol=tol)))

    @staticmethod
    def matrix_transpose(
        a: Tensor | TensorOperand,
        name: str = "matrix_transpose",
        conjugate: bool = False,
    ) -> Tensor:
        """Transpose the innermost two dimensions of `a`.

        Args:
            a: Input tensor of rank `>= 2`.
            name: Optional operation name.
            conjugate: If `True`, also compute complex conjugate.

        Returns:
            Tensor: Transposed tensor.
        """
        del name
        if conjugate:
            return linalg.adjoint(a)
        res = as_operand(_msc_ops.matrix_transpose(_to_tensor(a)))
        return _wrap_tensor(res)

    @staticmethod
    def matvec(
        a: Tensor | TensorOperand,
        b: Tensor | TensorOperand,
        transpose_a: bool = False,
        adjoint_a: bool = False,
        a_is_sparse: bool = False,
        b_is_sparse: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Multiply matrix `a` by vector `b`, producing `a @ b`.

        Args:
            a: Matrix tensor of shape `[..., M, N]`.
            b: Vector tensor of shape `[..., N]`.
            transpose_a: If `True`, transpose `a` before multiplication.
            adjoint_a: If `True`, conjugate-transpose `a` before multiplication.
            a_is_sparse: Hint that `a` is sparse.
            b_is_sparse: Hint that `b` is sparse.
            name: Optional operation name.

        Returns:
            Tensor: Matrix-vector product tensor of shape `[..., M]`.
        """
        del a_is_sparse, b_is_sparse, name
        b_col = _ops.expand_dims(_to_tensor(b), -1)
        prod = _to_tensor(linalg.matmul(a, b_col, transpose_a=transpose_a, adjoint_a=adjoint_a))
        return _wrap_tensor(_ops.squeeze(prod, axis=-1))

    @staticmethod
    def norm(
        tensor: Tensor | TensorOperand,
        ord: str | int | float = "euclidean",
        axis: int | Sequence[int] | None = None,
        keepdims: bool | None = None,
        name: str | None = None,
    ) -> Tensor:
        """Compute the norm of vectors, matrices, and tensors.

        Args:
            tensor: Input tensor.
            ord: Order of the norm (`'euclidean'`, `'fro'`, `1`, `2`, `np.inf`).
            axis: Dimension(s) along which to compute the norm.
            keepdims: If `True`, retains reduced dimensions with length `1`.
            name: Optional operation name.

        Returns:
            Tensor: Norm tensor.
        """
        del name
        t_in = _to_tensor(tensor)
        axis_val: int | builtins.tuple[int, ...] | None = (
            axis
            if isinstance(axis, int)
            else (
                builtins.tuple(int(x) for x in axis)
                if isinstance(axis, (builtins.list, builtins.tuple))
                else None
            )
        )
        kd_val = builtins.bool(keepdims)
        if ord in ("euclidean", "fro", 2, 2.0):
            res = _ops.sqrt(_to_tensor(_ops.sum(_ops.square(t_in), axis=axis_val, keepdims=kd_val)))
        elif ord in (1, 1.0):
            res = _to_tensor(_ops.sum(_ops.abs(t_in), axis=axis_val, keepdims=kd_val))
        elif ord == float("inf") or ord == "inf":
            res = _to_tensor(_ops.max(_ops.abs(t_in), axis=axis_val, keepdims=kd_val))
        elif ord == float("-inf") or ord == "-inf":
            res = _to_tensor(_ops.min(_ops.abs(t_in), axis=axis_val, keepdims=kd_val))
        else:
            p_f = float(ord)
            p_t = _to_tensor(p_f, dtype=t_in.dtype)
            inv_p_t = _to_tensor(1.0 / p_f, dtype=t_in.dtype)
            sum_p = _to_tensor(
                _ops.sum(_ops.power(_ops.abs(t_in), p_t), axis=axis_val, keepdims=kd_val)
            )
            res = _ops.power(sum_p, inv_p_t)
        return _wrap_tensor(res)

    @staticmethod
    def normalize(
        tensor: Tensor | TensorOperand,
        ord: str | int | float = "euclidean",
        axis: int | Sequence[int] | None = None,
        name: str | None = None,
    ) -> builtins.tuple[Tensor, Tensor]:
        """Normalize `tensor` along dimension `axis` using specified norm `ord`.

        Args:
            tensor: Input tensor.
            ord: Order of the norm (`'euclidean'`, `1`, `2`, `np.inf`).
            axis: Dimension(s) along which to normalize.
            name: Optional operation name.

        Returns:
            tuple[Tensor, Tensor]: Pair `(normalized, norm)` of normalized tensor and norm.
        """
        del name
        t_in = Tensor(tensor)
        n_val = linalg.norm(t_in, ord=ord, axis=axis, keepdims=axis is not None)
        return (t_in / n_val, n_val)

    @staticmethod
    def pinv(
        a: Tensor | TensorOperand,
        rcond: float | Tensor | None = None,
        validate_args: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Compute the Moore-Penrose pseudo-inverse of one or more matrices.

        Args:
            a: Matrix tensor.
            rcond: Cutoff for small singular values.
            validate_args: Whether to validate arguments.
            name: Optional operation name.

        Returns:
            Tensor: Pseudo-inverse tensor.
        """
        del rcond, validate_args, name
        return _wrap_tensor(_ops.pinv(_to_tensor(a)))

    @staticmethod
    def qr(
        input: Tensor | TensorOperand,
        full_matrices: bool = False,
        name: str | None = None,
    ) -> builtins.tuple[Tensor, Tensor]:
        """Compute the QR decomposition of one or more matrices.

        Args:
            input: Matrix tensor of shape `[..., M, N]`.
            full_matrices: If `True`, compute full-sized `Q` (`[..., M, M]`) and `R` (`[..., M, N]`).
            name: Optional operation name.

        Returns:
            tuple[Tensor, Tensor]: Pair `(q, r)` of orthogonal and upper-triangular factors.
        """
        del name
        t_in = Tensor(input)
        mode_str = "complete" if full_matrices else "reduced"
        if _tracer.is_tracing or len(t_in.shape) <= 2:
            q_raw, r_raw = _ops.qr(_to_tensor(t_in), mode=mode_str)
            return (_wrap_tensor(q_raw), _wrap_tensor(r_raw))
        batch_shape = t_in.shape[:-2]
        m, n = t_in.shape[-2], t_in.shape[-1]
        k = m if full_matrices else min(m, n)
        q_slices: list[Tensor] = []
        r_slices: list[Tensor] = []
        for idx in _iter_batch_indices(batch_shape):
            q_s, r_s = _ops.qr(_to_tensor(_slice_batch_matrix(t_in, idx)), mode=mode_str)
            q_slices.append(_wrap_tensor(q_s))
            r_slices.append(_wrap_tensor(r_s))
        return (
            _stack_batch_results(q_slices, batch_shape, (m, k), t_in.dtype),
            _stack_batch_results(r_slices, batch_shape, (k, n), t_in.dtype),
        )

    @staticmethod
    def set_diag(
        input: Tensor | TensorOperand,
        diagonal: Tensor | TensorOperand,
        name: str = "set_diag",
        k: int | builtins.tuple[int, int] = 0,
        align: str = "RIGHT_LEFT",
    ) -> Tensor:
        """Return a batched matrix tensor with new batched diagonal values.

        Args:
            input: Input matrix tensor of shape `[..., M, N]`.
            diagonal: Replacement diagonal values tensor.
            name: Optional operation name.
            k: Diagonal offset (`int` or `(k_low, k_high)` tuple).
            align: Alignment string.

        Returns:
            Tensor: Matrix tensor with updated diagonal(s).
        """
        del name, align
        t_in = Tensor(input)
        t_diag = Tensor(diagonal, dtype=t_in.dtype)
        if _tracer.is_tracing and isinstance(k, int) and k == 0 and len(t_in.shape) <= 2:
            raw_in = _to_tensor(t_in)
            old_diag = _to_tensor(
                linalg.diag(_wrap_tensor(as_operand(_msc_ops.tensor_diag_part(raw_in))))
            )
            new_diag = _to_tensor(linalg.diag(t_diag))
            return _wrap_tensor(_msc_ops.add(_msc_ops.subtract(raw_in, old_diag), new_diag))

        batch_shape = t_in.shape[:-2]
        m_rows = t_in.shape[-2]
        n_cols = t_in.shape[-1]
        if isinstance(k, builtins.tuple):
            k_low, k_high = int(k[0]), int(k[1])
        else:
            k_low = k_high = int(k)

        slices: list[Tensor] = []
        for idx in _iter_batch_indices(batch_shape):
            mat_2d = _to_nested_floats(_slice_batch_matrix(t_in, idx))
            diag_slice = t_diag
            for b_i in idx:
                diag_slice = diag_slice[b_i]
            if k_low == k_high:
                vals = _to_1d_floats(diag_slice)
                r_start = max(0, -k_low)
                c_start = max(0, k_low)
                for d_i, v in enumerate(vals):
                    mat_2d[r_start + d_i][c_start + d_i] = v
            else:
                band_rows = _to_nested_floats(diag_slice)
                for band_idx, offset in enumerate(range(k_high, k_low - 1, -1)):
                    r_start = max(0, -offset)
                    c_start = max(0, offset)
                    max_d = min(m_rows - r_start, n_cols - c_start, len(band_rows[band_idx]))
                    for d_i in range(max_d):
                        mat_2d[r_start + d_i][c_start + d_i] = band_rows[band_idx][d_i]
            slices.append(Tensor(mat_2d, dtype=t_in.dtype))
        if not batch_shape:
            return slices[0]
        return _stack_batch_results(slices, batch_shape, (m_rows, n_cols), t_in.dtype)

    @staticmethod
    def slogdet(
        input: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
        """Compute the sign and the log of the absolute value of the determinant.

        Args:
            input: Square matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor | tuple[Tensor, ...] | list[Tensor]: Pair `(sign, log_abs_determinant)`.
        """
        del name
        return _wrap(_ops.slogdet(_to_tensor(input)))

    @staticmethod
    def solve(
        matrix: Tensor | TensorOperand,
        rhs: Tensor | TensorOperand = None,
        adjoint: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Solve systems of linear equations `matrix @ X = rhs`.

        Args:
            matrix: Square invertible matrix tensor of shape `[..., M, M]`.
            rhs: Right-hand-side tensor of shape `[..., M, K]`.
            adjoint: If `True`, solve with the conjugate transpose of `matrix`.
            name: Optional operation name.

        Returns:
            Tensor: Solution tensor `X` of shape `[..., M, K]`.
        """
        del name
        t_lhs = linalg.adjoint(matrix) if adjoint else Tensor(matrix)
        t_rhs = t_lhs if rhs is None else Tensor(rhs, dtype=t_lhs.dtype)
        if _tracer.is_tracing or len(t_lhs.shape) <= 2:
            res = _ops.solve(_to_tensor(t_lhs), _to_tensor(t_rhs))
            return _wrap_tensor(res)
        batch_shape = t_lhs.shape[:-2]
        out_inner = t_rhs.shape[-2:]
        slices = [
            _wrap_tensor(
                _ops.solve(
                    _to_tensor(_slice_batch_matrix(t_lhs, idx)),
                    _to_tensor(_slice_batch_matrix(t_rhs, idx)),
                )
            )
            for idx in _iter_batch_indices(batch_shape)
        ]
        return _stack_batch_results(slices, batch_shape, out_inner, t_lhs.dtype)

    @staticmethod
    def sqrtm(
        input: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the matrix square root of one or more square matrices via Denman-Beavers / spectral iteration.

        Args:
            input: Square matrix tensor of shape `[..., M, M]`.
            name: Optional operation name.

        Returns:
            Tensor: Matrix square root tensor `S` such that `S @ S == input`.
        """
        del name
        t_in = Tensor(input)

        def _sqrtm_2d(mat_2d: Tensor) -> Tensor:
            """Compute matrix square root of a 2-D matrix.

            Args:
                mat_2d: 2-D square matrix `Tensor`.

            Returns:
                Tensor: 2-D matrix square root `Tensor`.
            """
            t_a = _to_tensor(mat_2d)
            vals, vecs = _ops.eigh(t_a)
            sqrt_vals = _ops.sqrt(_ops.maximum(vals, _to_tensor(0.0, dtype=mat_2d.dtype)))
            prod = _ops.matmul(
                _ops.matmul(vecs, _ops.diag(sqrt_vals)),
                _to_tensor(as_operand(_msc_ops.adjoint(vecs))),
            )
            return _wrap_tensor(prod)

        if _tracer.is_tracing or len(t_in.shape) <= 2:
            return _sqrtm_2d(t_in)
        batch_shape = t_in.shape[:-2]
        inner_shape = t_in.shape[-2:]
        slices = [
            _sqrtm_2d(_slice_batch_matrix(t_in, idx)) for idx in _iter_batch_indices(batch_shape)
        ]
        return _stack_batch_results(slices, batch_shape, inner_shape, t_in.dtype)

    @staticmethod
    def svd(
        tensor: Tensor | TensorOperand,
        full_matrices: bool = False,
        compute_uv: bool = True,
        name: str | None = None,
    ) -> Tensor | builtins.tuple[Tensor, ...]:
        """Compute the singular value decompositions of one or more matrices.

        Args:
            tensor: Matrix tensor of shape `[..., M, N]`.
            full_matrices: If `True`, compute full-sized `u` and `v`.
            compute_uv: If `True`, compute left and right singular vectors `(s, u, v)`.
            name: Optional operation name.

        Returns:
            Tensor | tuple[Tensor, ...]: Singular values `s` or `(s, u, v)`.
        """
        del name
        t_in = Tensor(tensor)
        if _tracer.is_tracing or len(t_in.shape) <= 2:
            res = _ops.svd(_to_tensor(t_in), full_matrices=full_matrices, compute_uv=compute_uv)
            if not compute_uv:
                return _wrap_tensor(res)
            u_raw, s_raw, v_raw = res[0], res[1], res[2]
            # TensorFlow's tf.linalg.svd returns (s, u, v) where A = u @ diag(s) @ v^T
            return (
                _wrap_tensor(s_raw),
                _wrap_tensor(u_raw),
                linalg.adjoint(_wrap_tensor(v_raw)),
            )

        batch_shape = t_in.shape[:-2]
        m, n = t_in.shape[-2], t_in.shape[-1]
        k = min(m, n)
        s_slices: list[Tensor] = []
        u_slices: list[Tensor] = []
        v_slices: list[Tensor] = []
        for idx in _iter_batch_indices(batch_shape):
            sub = _ops.svd(
                _to_tensor(_slice_batch_matrix(t_in, idx)),
                full_matrices=full_matrices,
                compute_uv=compute_uv,
            )
            if not compute_uv:
                s_slices.append(_wrap_tensor(sub))
            else:
                s_slices.append(_wrap_tensor(sub[1]))
                u_slices.append(_wrap_tensor(sub[0]))
                v_slices.append(linalg.adjoint(_wrap_tensor(sub[2])))
        s_batched = _stack_batch_results(s_slices, batch_shape, (k,), t_in.dtype)
        if not compute_uv:
            return s_batched
        u_inner = (m, m) if full_matrices else (m, k)
        v_inner = (n, n) if full_matrices else (n, k)
        return (
            s_batched,
            _stack_batch_results(u_slices, batch_shape, u_inner, t_in.dtype),
            _stack_batch_results(v_slices, batch_shape, v_inner, t_in.dtype),
        )

    @staticmethod
    def tensor_diag(
        diagonal: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Return a diagonal tensor with a given diagonal values vector.

        Args:
            diagonal: 1-D tensor of diagonal values.
            name: Optional operation name.

        Returns:
            Tensor: 2-D diagonal matrix tensor.
        """
        del name
        return _wrap_tensor(_ops.diag(_to_tensor(diagonal)))

    @staticmethod
    def tensor_diag_part(
        input: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Return the diagonal part of a tensor.

        Args:
            input: Rank-2k tensor.
            name: Optional operation name.

        Returns:
            Tensor: Extracted diagonal tensor.
        """
        del name
        return _wrap_tensor(as_operand(_msc_ops.tensor_diag_part(_to_tensor(input))))

    @staticmethod
    def tensordot(
        a: Tensor | TensorOperand,
        b: Tensor | TensorOperand,
        axes: int | Sequence[int] | Sequence[Sequence[int]],
        name: str | None = None,
    ) -> Tensor:
        """Tensor contraction of `a` and `b` along specified `axes`.

        Args:
            a: First `Tensor`.
            b: Second `Tensor`.
            axes: Scalar integer or sequence of axes to contract.
            name: Optional operation name.

        Returns:
            Tensor: Contracted tensor.
        """
        del name
        res = _ops.tensordot(_to_tensor(a), _to_tensor(b), axes=axes)
        return _wrap_tensor(res)

    @staticmethod
    def trace(
        x: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the trace of a [batch of] matrix `x` (sum along the main diagonal).

        Args:
            x: Input matrix tensor of shape `[..., M, N]`.
            name: Optional operation name.

        Returns:
            Tensor: Trace tensor of shape `[...]`.
        """
        del name
        t_x = Tensor(x)
        if _tracer.is_tracing or len(t_x.shape) <= 2:
            return _wrap_tensor(_ops.trace(_to_tensor(t_x)))
        diag_v = linalg.diag_part(t_x)
        return _wrap_tensor(_ops.sum(_to_tensor(diag_v), axis=-1))

    @staticmethod
    def triangular_solve(
        matrix: Tensor | TensorOperand,
        rhs: Tensor | TensorOperand,
        lower: bool = True,
        adjoint: bool = False,
        name: str | None = None,
    ) -> Tensor:
        """Solve systems of linear equations with lower or upper triangular matrices.

        Args:
            matrix: Triangular coefficient matrix tensor of shape `[..., M, M]`.
            rhs: Right-hand side tensor of shape `[..., M, K]`.
            lower: If `True`, `matrix` is lower triangular; otherwise upper triangular.
            adjoint: If `True`, solve with the conjugate transpose of `matrix`.
            name: Optional operation name.

        Returns:
            Tensor: Solution tensor of shape `[..., M, K]`.
        """
        del name
        t_mat = Tensor(matrix)
        t_rhs = Tensor(rhs, dtype=t_mat.dtype)
        if _tracer.is_tracing or len(t_mat.shape) <= 2:
            return _wrap_tensor(
                as_operand(
                    _msc_ops.triangular_solve(
                        _to_tensor(t_mat),
                        _to_tensor(t_rhs),
                        lower=lower,
                        adjoint=adjoint,
                    )
                )
            )
        batch_shape = t_mat.shape[:-2]
        out_inner = t_rhs.shape[-2:]
        slices = [
            _wrap_tensor(
                as_operand(
                    _msc_ops.triangular_solve(
                        _to_tensor(_slice_batch_matrix(t_mat, idx)),
                        _to_tensor(_slice_batch_matrix(t_rhs, idx)),
                        lower=lower,
                        adjoint=adjoint,
                    )
                )
            )
            for idx in _iter_batch_indices(batch_shape)
        ]
        return _stack_batch_results(slices, batch_shape, out_inner, t_mat.dtype)

    @staticmethod
    def _build_tridiagonal_matrix(
        diagonals: Tensor | TensorOperand,
        diagonals_format: str = "compact",
    ) -> Tensor:
        """Build a dense square tridiagonal matrix from `diagonals`.

        Args:
            diagonals: Sequence `(superdiag, maindiag, subdiag)`, compact `[3, M]`, or matrix.
            diagonals_format: Format of `diagonals` (`'compact'`, `'sequence'`, or `'matrix'`).

        Returns:
            Tensor: Dense `[M, M]` tridiagonal matrix tensor.
        """
        if diagonals_format == "matrix":
            return Tensor(diagonals)
        if isinstance(diagonals, (builtins.tuple, builtins.list)) and len(diagonals) == 3:
            du = _to_tensor(diagonals[0])
            d = _to_tensor(diagonals[1])
            dl = _to_tensor(diagonals[2])
        else:
            t_diag = Tensor(diagonals)
            du = _to_tensor(t_diag[0])
            d = _to_tensor(t_diag[1])
            dl = _to_tensor(t_diag[2])
        main_m = _to_tensor(linalg.diag(d))
        super_m = _to_tensor(
            linalg.band_part(_ops.roll(_to_tensor(linalg.diag(du)), shift=1, axis=-1), 0, 1)
        )
        sub_m = _to_tensor(
            linalg.band_part(_ops.roll(_to_tensor(linalg.diag(dl)), shift=-1, axis=-1), 1, 0)
        )
        return _wrap_tensor(_msc_ops.add(_msc_ops.add(main_m, super_m), sub_m))

    @staticmethod
    def tridiagonal_matmul(
        diagonals: Tensor | TensorOperand,
        rhs: Tensor | TensorOperand,
        diagonals_format: str = "compact",
        name: str | None = None,
    ) -> Tensor:
        """Multiply a tridiagonal matrix by a right-hand-side matrix `rhs`.

        Args:
            diagonals: Sequence `(superdiag, maindiag, subdiag)` or compact `[3, M]` tensor.
            rhs: Right-hand-side tensor of shape `[..., M, N]`.
            diagonals_format: Format of `diagonals` (`'compact'`, `'sequence'`, or `'matrix'`).
            name: Optional operation name.

        Returns:
            Tensor: Product tensor.
        """
        del name
        mat = linalg._build_tridiagonal_matrix(diagonals, diagonals_format=diagonals_format)
        return linalg.matmul(mat, rhs)

    @staticmethod
    def tridiagonal_solve(
        diagonals: Tensor | TensorOperand,
        rhs: Tensor | TensorOperand,
        diagonals_format: str = "compact",
        transpose_rhs: bool = False,
        conjugate_rhs: bool = False,
        name: str | None = None,
        partial_pivoting: bool = True,
        perturb_singular: bool = False,
    ) -> Tensor:
        """Solve a tridiagonal system of linear equations.

        Args:
            diagonals: Tridiagonal representation (`'compact'`, `'sequence'`, or `'matrix'`).
            rhs: Right-hand side tensor.
            diagonals_format: Format of `diagonals`.
            transpose_rhs: Whether to transpose `rhs` before solving.
            conjugate_rhs: Whether to conjugate `rhs` before solving.
            name: Optional operation name.
            partial_pivoting: Whether to use partial pivoting.
            perturb_singular: Whether to perturb singular pivots.

        Returns:
            Tensor: Solution tensor.
        """
        del transpose_rhs, conjugate_rhs, name, partial_pivoting, perturb_singular
        mat = linalg._build_tridiagonal_matrix(diagonals, diagonals_format=diagonals_format)
        return linalg.solve(mat, rhs)
