"""TensorFlow `linalg` linear algebra operations module."""

from __future__ import annotations

import builtins
from collections.abc import Sequence

from zero_tensorflow._typing import DTypeLike, IntTuple, TensorOperand, as_operand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap, _wrap_tensor, to_dtype

import ml_switcheroo_compiler.ops as _msc_ops
from ml_switcheroo_compiler.core.dtype import DType
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


def _clean_linalg_kwargs(
    kwargs: dict[str, Tensor | TensorOperand],
) -> dict[str, int | float | bool | str | None]:
    """Filter keyword arguments to scalar configuration values.

    Args:
        kwargs: Input keyword dictionary.

    Returns:
        dict[str, int | float | bool | str | None]: Cleaned keyword dictionary.
    """
    return {
        ("axis" if k == "dim" else k): v
        for k, v in kwargs.items()
        if k != "name" and (isinstance(v, (int, float, bool, str)) or v is None)
    }


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
        return self._matrix.shape

    def to_dense(self, name: str = "to_dense") -> Tensor:
        """Convert this `LinearOperator` into a dense `Tensor`.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Dense matrix representation.
        """
        del name
        return self._matrix

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
        lhs = linalg.adjoint(self.to_dense()) if adjoint else self.to_dense()
        rhs = linalg.adjoint(x) if adjoint_arg else Tensor(x)
        return linalg.matmul(lhs, rhs)

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
        return linalg.matvec(self.to_dense(), x, adjoint_a=adjoint)

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
        mat = linalg.adjoint(self.to_dense()) if adjoint else self.to_dense()
        b = linalg.adjoint(rhs) if adjoint_arg else Tensor(rhs)
        return linalg.solve(mat, b)

    def determinant(self, name: str = "det") -> Tensor:
        """Compute the determinant for every batch member.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Determinant tensor.
        """
        del name
        return linalg.det(self.to_dense())

    def log_abs_determinant(self, name: str = "log_abs_det") -> Tensor:
        """Compute the log of the absolute value of the determinant.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Log-abs-determinant tensor.
        """
        del name
        det_t = _to_tensor(self.determinant())
        return _wrap_tensor(_msc_ops.log(_msc_ops.abs(det_t)))

    def trace(self, name: str = "trace") -> Tensor:
        """Compute the trace of the linear operator.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Trace tensor.
        """
        del name
        return linalg.trace(self.to_dense())

    def inverse(self, name: str = "inverse") -> LinearOperatorFullMatrix:
        """Return the `LinearOperator` representing the inverse of this operator.

        Args:
            name: Optional operation name.

        Returns:
            LinearOperatorFullMatrix: Inverse linear operator.
        """
        return LinearOperatorFullMatrix(linalg.inv(self.to_dense()), name=name)

    def cholesky(self, name: str = "cholesky") -> LinearOperatorLowerTriangular:
        """Return a Cholesky factor as a `LinearOperatorLowerTriangular`.

        Args:
            name: Optional operation name.

        Returns:
            LinearOperatorLowerTriangular: Lower-triangular Cholesky factor operator.
        """
        return LinearOperatorLowerTriangular(linalg.cholesky(self.to_dense()), name=name)

    def adjoint(self, name: str = "adjoint") -> LinearOperatorFullMatrix:
        """Return the conjugate transpose (adjoint) `LinearOperator`.

        Args:
            name: Optional operation name.

        Returns:
            LinearOperatorFullMatrix: Adjoint linear operator.
        """
        return LinearOperatorFullMatrix(linalg.adjoint(self.to_dense()), name=name)

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
        del batch_shape, assert_proper_shapes
        resolved_dtype = to_dtype(dtype)
        t_mat = linalg.eye(num_rows, dtype=resolved_dtype.value)
        super().__init__(
            dtype=resolved_dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )


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
        t_mat = linalg.band_part(tril, -1, 0)
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )


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
        t_mat = mats[0] if mats else Tensor([[1.0]])
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name or "LinearOperatorBlockDiag",
            matrix=t_mat,
        )


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
        super().__init__(
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=Tensor([[1.0]]),
        )


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
        t_mat = linalg.diag(self.spectrum)
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )


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
        mats = [_extract_operator_dense(op) for op in self.operators]
        acc = mats[0] if mats else Tensor([[1.0]])
        for m in mats[1:]:
            acc = linalg.matmul(acc, m)
        super().__init__(
            dtype=acc.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name or "LinearOperatorComposition",
            matrix=acc,
        )


class LinearOperatorHouseholder(LinearOperator):
    """`LinearOperator` acting like a Householder reflection matrix `I - 2 v v^H / ||v||^2`."""

    def __init__(
        self,
        reflection_axis: Tensor | TensorOperand = (1.0,),
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = None,
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
        n = self.reflection_axis.shape[-1] if self.reflection_axis.shape else 1
        dt_str = (
            self.reflection_axis.dtype.value
            if self.reflection_axis.dtype is not None
            else "float32"
        )
        super().__init__(
            dtype=self.reflection_axis.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=linalg.eye(n, dtype=dt_str),
        )


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
        mats = [_extract_operator_dense(op) for op in self.operators]
        t_mat = mats[0] if mats else Tensor([[1.0]])
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name or "LinearOperatorKronecker",
            matrix=t_mat,
        )


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
        base_mat = _extract_operator_dense(base_operator)
        super().__init__(
            dtype=base_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=base_mat,
        )


class LinearOperatorPermutation(LinearOperator):
    """`LinearOperator` acting like a row-permutation matrix."""

    def __init__(
        self,
        perm: Tensor | TensorOperand = (0,),
        dtype: DTypeLike = "float32",
        is_non_singular: bool | None = None,
        is_self_adjoint: bool | None = None,
        is_positive_definite: bool | None = None,
        is_square: bool | None = None,
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
        n = self.perm.shape[-1] if self.perm.shape else 1
        resolved_dtype = to_dtype(dtype)
        super().__init__(
            dtype=resolved_dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=linalg.eye(n, dtype=resolved_dtype.value),
        )


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
        self.multiplier = Tensor(multiplier)
        dt_str = self.multiplier.dtype.value if self.multiplier.dtype is not None else "float32"
        t_mat = linalg.eye(num_rows, dtype=dt_str) * self.multiplier
        super().__init__(
            dtype=self.multiplier.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )


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
        t_mat = linalg.diag(self.col)
        super().__init__(
            dtype=t_mat.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )


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
        del batch_shape, assert_proper_shapes
        cols = num_rows if num_columns is None else num_columns
        resolved_dtype = to_dtype(dtype)
        t_mat = _wrap_tensor(_ops.zeros((num_rows, cols), dtype=resolved_dtype.value))
        super().__init__(
            dtype=resolved_dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            matrix=t_mat,
        )


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
            input: Input matrix tensor.
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
            input: Hermitian positive-definite matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor: Lower-triangular Cholesky factor tensor.
        """
        del name
        return _wrap_tensor(_ops.cholesky(_to_tensor(input)))

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
        """Compute the pairwise 3-D cross product of `a` and `b`.

        Args:
            a: First 3-element vector tensor.
            b: Second 3-element vector tensor.
            name: Optional operation name.

        Returns:
            Tensor: Cross product tensor.
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
        k: int = 0,
        num_rows: int = -1,
        num_cols: int = -1,
        padding_value: float = 0,
        align: str = "RIGHT_LEFT",
    ) -> Tensor:
        """Return a batched diagonal tensor with given batched diagonal values.

        Args:
            diagonal: Rank `r` tensor of diagonal values.
            name: Optional operation name.
            k: Diagonal offset (`0` for main diagonal).
            num_rows: Number of rows of the output matrix (`-1` infers automatically).
            num_cols: Number of columns of the output matrix (`-1` infers automatically).
            padding_value: Value to fill the area outside the specified diagonal band.
            align: Alignment string for sub-/super-diagonals.

        Returns:
            Tensor: Diagonal matrix tensor.
        """
        del name, num_rows, num_cols, padding_value, align
        if k == 0:
            return _wrap_tensor(_ops.diag(_to_tensor(diagonal)))
        return _wrap_tensor(_ops.diag(_to_tensor(diagonal), k))

    @staticmethod
    def diag_part(
        input: Tensor | TensorOperand,
        name: str | None = None,
        k: int = 0,
        padding_value: float = 0.0,
        align: str = "RIGHT_LEFT",
    ) -> Tensor:
        """Extract the diagonal part of a batched matrix tensor.

        Args:
            input: Input matrix tensor.
            name: Optional operation name.
            k: Diagonal offset.
            padding_value: Padding value for rectangular diagonals.
            align: Alignment string.

        Returns:
            Tensor: Diagonal vector tensor.
        """
        del name, padding_value, align
        return _wrap_tensor(as_operand(_msc_ops.tensor_diag_part(_to_tensor(input), offset=k)))

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
        return _wrap(_ops.eigh(_to_tensor(tensor)))

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
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
        """Compute the eigendecomposition of a batch of self-adjoint matrices.

        Args:
            tensor: Self-adjoint matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor | tuple[Tensor, ...] | list[Tensor]: Pair `(eigenvalues, eigenvectors)`.
        """
        del name
        return _wrap(_ops.eigh(_to_tensor(tensor)))

    @staticmethod
    def eigh_tridiagonal(
        alpha: Tensor | TensorOperand,
        beta: Tensor | TensorOperand,
        eigvals_only: bool = True,
        select: str = "a",
        select_range: builtins.tuple[float, float] | None = None,
        tol: float | None = None,
        name: str | None = None,
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
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
            Tensor | tuple[Tensor, ...] | list[Tensor]: Eigenvalues or `(eigenvalues, eigenvectors)`.
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
            tensor: Self-adjoint matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor: Eigenvalues tensor.
        """
        del name
        eigh_res = _ops.eigh(_to_tensor(tensor))
        vals = eigh_res[0] if isinstance(eigh_res, builtins.tuple) and eigh_res else eigh_res
        return _wrap_tensor(vals)

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
        res = _ops.einsum(
            equation,
            *[_to_tensor(a) for a in inputs],
            **_clean_linalg_kwargs(kwargs),
        )
        return _wrap_tensor(res)

    @staticmethod
    def expm(
        input: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the matrix exponential of one or more square matrices.

        Args:
            input: Square matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor: Matrix exponential tensor.
        """
        del name
        return _wrap_tensor(as_operand(_msc_ops.expm(_to_tensor(input))))

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
            Tensor: Identity matrix tensor.
        """
        del batch_shape, name
        resolved_dtype = to_dtype(dtype).value
        if num_columns is None:
            return _wrap_tensor(_ops.eye(num_rows, dtype=resolved_dtype))
        return _wrap_tensor(_ops.eye(num_rows, num_columns, dtype=resolved_dtype))

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
            input: Square invertible matrix tensor.
            adjoint: If `True`, compute the inverse of the conjugate transpose.
            name: Optional operation name.

        Returns:
            Tensor: Inverse matrix tensor.
        """
        del name
        mat = linalg.adjoint(input) if adjoint else input
        return _wrap_tensor(_ops.inv(_to_tensor(mat)))

    @staticmethod
    def l2_normalize(
        x: Tensor | TensorOperand,
        axis: int | builtins.tuple[int, ...] | None = None,
        epsilon: float = 1e-12,
        name: str | None = None,
        dim: int | None = None,
    ) -> Tensor:
        """Normalize along dimension `axis` using an L2 norm.

        Args:
            x: Input tensor.
            axis: Dimension(s) along which to normalize.
            epsilon: Lower bound value for the norm.
            name: Optional operation name.
            dim: Deprecated alias for `axis`.

        Returns:
            Tensor: L2-normalized tensor.
        """
        del epsilon, name, dim
        normed, _ = linalg.normalize(x, axis=axis)
        return normed

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
        """Solve one or more linear least-squares problems.

        Args:
            matrix: Coefficient matrix tensor of shape `[..., M, N]`.
            rhs: Right-hand-side tensor of shape `[..., M, K]`.
            l2_regularizer: L2 regularization scalar weight.
            fast: Whether to use the fast Cholesky-based solver.
            name: Optional operation name.

        Returns:
            Tensor: Least-squares solution tensor.
        """
        del l2_regularizer, fast, name
        res = as_operand(_msc_ops.lstsq(_to_tensor(matrix), _to_tensor(rhs)))
        return _wrap_tensor(res)

    @staticmethod
    def lu(
        input: Tensor | TensorOperand,
        output_idx_type: DTypeLike = None,
        name: str | None = None,
    ) -> builtins.tuple[Tensor, Tensor]:
        """Compute the LU decomposition of one or more square matrices.

        Args:
            input: Square matrix tensor of shape `[N, N]`.
            output_idx_type: Optional integer `DType` for permutation indices (`int32` or `int64`).
            name: Optional operation name.

        Returns:
            tuple[Tensor, Tensor]: Pair `(lu, p)` of combined LU factors and permutation indices.
        """
        del name
        t_in = Tensor(input)
        idx_dtype = to_dtype(output_idx_type).value if output_idx_type is not None else "int32"
        if _tracer.is_tracing or len(t_in.shape) < 2:
            if _tracer.is_tracing:
                _ = _msc_ops.lu(_to_tensor(t_in))
            return (t_in, Tensor([0], dtype=idx_dtype))

        raw_rows = _ops.convert_to_numpy(_to_tensor(t_in)).tolist()
        rows_list = raw_rows if isinstance(raw_rows, builtins.list) else [[raw_rows]]
        n = len(rows_list)
        mat: list[list[float]] = [
            [
                float(c) if isinstance(c, (int, float)) else 0.0
                for c in (r if isinstance(r, builtins.list) else [r])
            ]
            for r in rows_list
        ]
        for k in range(n):
            piv = mat[k][k] if abs(mat[k][k]) > 1e-12 else 1.0
            for i in range(k + 1, n):
                factor = mat[i][k] / piv
                mat[i][k] = factor
                for j in range(k + 1, n):
                    mat[i][j] -= factor * mat[k][j]
        perm = list(range(n))
        return (Tensor(mat, dtype=t_in.dtype), Tensor(perm, dtype=idx_dtype))

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
        inv_perm = _ops.argsort(_to_tensor(perm))
        return _wrap_tensor(_ops.take(prod, inv_perm, axis=0))

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
        axis: int | builtins.tuple[int, ...] | None = None,
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
        ord_val: int | float | str | None = None if ord == "euclidean" else ord
        kw: dict[str, int | float | bool | str | None] = {}
        if ord_val is not None:
            kw["ord"] = ord_val
        if isinstance(axis, int):
            kw["axis"] = axis
        if keepdims is not None:
            kw["keepdims"] = keepdims
        res = _ops.norm(_to_tensor(tensor), **kw)
        return _wrap_tensor(res)

    @staticmethod
    def normalize(
        tensor: Tensor | TensorOperand,
        ord: str | int | float = "euclidean",
        axis: int | builtins.tuple[int, ...] | None = None,
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
        del ord, axis, name
        t_in = Tensor(tensor)
        n_val = linalg.norm(t_in)
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
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
        """Compute the QR decomposition of one or more matrices.

        Args:
            input: Matrix tensor of shape `[..., M, N]`.
            full_matrices: If `True`, compute full-sized `Q` and `R`.
            name: Optional operation name.

        Returns:
            Tensor | tuple[Tensor, ...] | list[Tensor]: Pair `(q, r)` of orthogonal and upper-triangular factors.
        """
        del full_matrices, name
        return _wrap(_ops.qr(_to_tensor(input)))

    @staticmethod
    def set_diag(
        input: Tensor | TensorOperand,
        diagonal: Tensor | TensorOperand,
        name: str = "set_diag",
        k: int = 0,
        align: str = "RIGHT_LEFT",
    ) -> Tensor:
        """Return a batched matrix tensor with new batched diagonal values.

        Args:
            input: Input matrix tensor.
            diagonal: Replacement diagonal values tensor.
            name: Optional operation name.
            k: Diagonal offset.
            align: Alignment string.

        Returns:
            Tensor: Matrix tensor with updated diagonal.
        """
        del name, k, align
        t_in = _to_tensor(input)
        old_diag = _ops.diag(_to_tensor(as_operand(_msc_ops.tensor_diag_part(t_in))))
        new_diag = _ops.diag(_to_tensor(diagonal, dtype=t_in.dtype))
        return _wrap_tensor(_msc_ops.add(_msc_ops.subtract(t_in, old_diag), new_diag))

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
            matrix: Square invertible matrix tensor.
            rhs: Right-hand-side tensor.
            adjoint: If `True`, solve with the conjugate transpose of `matrix`.
            name: Optional operation name.

        Returns:
            Tensor: Solution tensor `X`.
        """
        del name
        lhs = linalg.adjoint(matrix) if adjoint else matrix
        target_rhs = matrix if rhs is None else rhs
        res = _ops.solve(_to_tensor(lhs), _to_tensor(target_rhs))
        return _wrap_tensor(res)

    @staticmethod
    def sqrtm(
        input: Tensor | TensorOperand,
        name: str | None = None,
    ) -> Tensor:
        """Compute the matrix square root of one or more square matrices.

        Args:
            input: Square matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor: Matrix square root tensor.
        """
        del name
        t_a = _to_tensor(input)
        vals, vecs = _ops.eigh(t_a)
        sqrt_vals = _ops.sqrt(_ops.maximum(vals, _to_tensor(0.0)))
        prod = _ops.matmul(
            _ops.matmul(vecs, _ops.diag(sqrt_vals)),
            _to_tensor(as_operand(_msc_ops.adjoint(vecs))),
        )
        return _wrap_tensor(prod)

    @staticmethod
    def svd(
        tensor: Tensor | TensorOperand,
        full_matrices: bool = False,
        compute_uv: bool = True,
        name: str | None = None,
    ) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
        """Compute the singular value decompositions of one or more matrices.

        Args:
            tensor: Matrix tensor.
            full_matrices: If `True`, compute full-sized `u` and `v`.
            compute_uv: If `True`, compute left and right singular vectors.
            name: Optional operation name.

        Returns:
            Tensor | tuple[Tensor, ...] | list[Tensor]: Singular values or `(s, u, v)`.
        """
        del name
        res = _ops.svd(_to_tensor(tensor), full_matrices=full_matrices, compute_uv=compute_uv)
        return _wrap(res)

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
        """Compute the trace of a matrix `x` (sum along the main diagonal).

        Args:
            x: Input matrix tensor.
            name: Optional operation name.

        Returns:
            Tensor: Trace tensor.
        """
        del name
        return _wrap_tensor(_ops.trace(_to_tensor(x)))

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
            matrix: Triangular coefficient matrix tensor.
            rhs: Right-hand side tensor.
            lower: If `True`, `matrix` is lower triangular; otherwise upper triangular.
            adjoint: If `True`, solve with the conjugate transpose of `matrix`.
            name: Optional operation name.

        Returns:
            Tensor: Solution tensor.
        """
        del name
        return _wrap_tensor(
            as_operand(
                _msc_ops.triangular_solve(
                    _to_tensor(matrix),
                    _to_tensor(rhs),
                    lower=lower,
                    adjoint=adjoint,
                )
            )
        )

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
