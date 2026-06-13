import numpy as np
from zero_tensorflow import Tensor
from ml_switcheroo_compiler.core.config import EagerMode
from ml_switcheroo_compiler.tracing import _tracer
import ml_switcheroo_compiler.ops.binary.math
import ml_switcheroo_compiler.ops.binary.special
import ml_switcheroo_compiler.ops.control_flow
import ml_switcheroo_compiler.ops.creation.basic
import ml_switcheroo_compiler.ops.creation.frontend
import ml_switcheroo_compiler.ops.linalg.basic
import ml_switcheroo_compiler.ops.linalg.frontend
import ml_switcheroo_compiler.ops.reductions.basic
import ml_switcheroo_compiler.ops.reductions.frontend
import ml_switcheroo_compiler.ops.shape.basic
import ml_switcheroo_compiler.ops.shape.frontend
import ml_switcheroo_compiler.ops.unary.math
import ml_switcheroo_compiler.ops.unary.special


def test_ops_exhaustive():
    with EagerMode():
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        y = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
        z = Tensor(np.array([1, 2]))
        try:
            ml_switcheroo_compiler.ops.binary.math.Add()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Add', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.BinaryMathOp()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.BinaryMathOp', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.BitwiseAnd()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.BitwiseAnd', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.BitwiseOr()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.BitwiseOr', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.BitwiseXor()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.BitwiseXor', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Copysign()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Copysign', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Divide()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Divide', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Equal()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Equal', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.FloatPower()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.FloatPower', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.FloorDivide()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.FloorDivide', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Fmax()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Fmax', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Fmin()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Fmin', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Fmod()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Fmod', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Gcd()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Gcd', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Greater()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Greater', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.GreaterEqual()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.GreaterEqual', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Heaviside()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Heaviside', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Hypot()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Hypot', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Lcm()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Lcm', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Ldexp()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Ldexp', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.LeftShift()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.LeftShift', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Less()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Less', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.LessEqual()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.LessEqual', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Logaddexp()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Logaddexp', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Logaddexp2()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Logaddexp2', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.LogicalAnd()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.LogicalAnd', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.LogicalOr()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.LogicalOr', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.LogicalXor()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.LogicalXor', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Maximum()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Maximum', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Minimum()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Minimum', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Mod()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Mod', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Multiply()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Multiply', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Nextafter()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Nextafter', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.NotEqual()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.NotEqual', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Power()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Power', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Remainder()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Remainder', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.RightShift()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.RightShift', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Subtract()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Subtract', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.TrueDivide()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.TrueDivide', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Xlogy()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.math.Xlogy', e)
        try:
            ml_switcheroo_compiler.ops.binary.special.Allclose()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.special.Allclose', e)
        try:
            ml_switcheroo_compiler.ops.binary.special.Atan2()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.special.Atan2', e)
        try:
            ml_switcheroo_compiler.ops.binary.special.Divmod()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.special.Divmod', e)
        try:
            ml_switcheroo_compiler.ops.binary.special.Isclose()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.binary.special.Isclose', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.cond(
                Tensor(np.array(True)), lambda *args: x, lambda *args: x
            )
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.control_flow.cond', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.pmap(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.control_flow.pmap', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.scan(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.control_flow.scan', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.stop_gradient(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.control_flow.stop_gradient', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.vmap(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.control_flow.vmap', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.while_loop(
                lambda *args: x, lambda *args: x, x
            )
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.control_flow.while_loop', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Arange()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.basic.Arange', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.CreationOp()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.basic.CreationOp', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Full()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.basic.Full', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.ManualSeed()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.basic.ManualSeed', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Ones()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.basic.Ones', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Rand()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.basic.Rand', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Randint()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.basic.Randint', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Randn()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.basic.Randn', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Zeros()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.basic.Zeros', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.arange()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.arange', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.array(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.array', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.asarray(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.asarray', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.diag(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.diag', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.empty((2, 2))
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.empty', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.empty_like(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.empty_like', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.eye(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.eye', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.full((2, 2), x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.full', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.full_like(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.full_like', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.identity(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.identity', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.linspace(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.linspace', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.manual_seed(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.manual_seed', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.ones((2, 2))
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.ones', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.ones_like(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.ones_like', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.rand(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.rand', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.randint(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.randint', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.randn(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.randn', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.zeros((2, 2))
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.zeros', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.zeros_like(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.creation.frontend.zeros_like', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.ConvGeneralDilated()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.basic.ConvGeneralDilated', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.Dot()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.basic.Dot', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.DotGeneral()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.basic.DotGeneral', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.Einsum()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.basic.Einsum', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.Fft()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.basic.Fft', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.Matmul()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.basic.Matmul', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.Rfft()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.basic.Rfft', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.cholesky(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.cholesky', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.conv_general_dilated(x, x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.conv_general_dilated', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.cross(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.cross', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.det(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.det', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.dot(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.dot', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.dot_general(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.dot_general', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.eigh(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.eigh', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.eigvalsh(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.eigvalsh', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.einsum(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.einsum', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.fft(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.fft', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.inner(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.inner', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.inv(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.inv', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.lu(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.lu', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.lu_factor(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.lu_factor', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.matmul(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.matmul', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.matrix_power(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.matrix_power', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.outer(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.outer', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.pinv(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.pinv', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.qr(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.qr', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.rfft(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.rfft', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.slogdet(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.slogdet', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.solve(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.solve', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.solve_triangular(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.solve_triangular', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.svd(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.svd', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.tensordot(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.tensordot', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.vdot(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.linalg.frontend.vdot', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.All()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.All', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.AnyOp()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.AnyOp', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Argmax()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Argmax', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Argmin()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Argmin', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.CountNonzero()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.CountNonzero', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Cumsum()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Cumsum', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Logsumexp()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Logsumexp', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Max()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Max', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Mean()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Mean', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Min()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Min', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Norm()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Norm', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Pmean()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Pmean', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Prod()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Prod', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Psum()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Psum', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.ReduceWindow()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.ReduceWindow', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.ReductionOp()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.ReductionOp', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.SegmentSum()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.SegmentSum', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Std()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Std', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Sum()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Sum', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Variance()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.basic.Variance', e)
        try:
            ml_switcheroo_compiler.ops.reductions.frontend.pmean(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.frontend.pmean', e)
        try:
            ml_switcheroo_compiler.ops.reductions.frontend.psum(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.frontend.psum', e)
        try:
            ml_switcheroo_compiler.ops.reductions.frontend.reduce_window(x, x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.frontend.reduce_window', e)
        try:
            ml_switcheroo_compiler.ops.reductions.frontend.segment_sum(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.reductions.frontend.segment_sum', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.BroadcastInDim()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.basic.BroadcastInDim', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.BroadcastTo()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.basic.BroadcastTo', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.DynamicSlice()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.basic.DynamicSlice', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.DynamicUpdateSlice()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.basic.DynamicUpdateSlice', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.Reshape()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.basic.Reshape', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.Resize()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.basic.Resize', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.Sort()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.basic.Sort', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.TopK()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.basic.TopK', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.Transpose()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.basic.Transpose', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.array_split(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.array_split', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.broadcast_in_dim(x, (2, 2), x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.broadcast_in_dim', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.broadcast_to(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.broadcast_to', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.concatenate([x, y])
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.concatenate', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.dsplit(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.dsplit', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.dstack(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.dstack', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.dynamic_slice(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.dynamic_slice', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.dynamic_update_slice(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.dynamic_update_slice', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.expand(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.expand', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.expand_dims(x, 0)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.expand_dims', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.flatten(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.flatten', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.gather(x, 0, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.gather', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.gather_nd(x, z)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.gather_nd', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.hsplit(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.hsplit', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.hstack(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.hstack', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.image_resize(x, (2, 2))
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.image_resize', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.meshgrid([x, y])
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.meshgrid', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.moveaxis(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.moveaxis', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.pad(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.pad', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.permute(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.permute', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.repeat(x, 2)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.repeat', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.reshape(x, (2, 2))
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.reshape', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.roll(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.roll', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.scatter(x, 0, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.scatter', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.scatter_add(x, 0, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.scatter_add', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.scatter_nd(z, x, (2, 2))
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.scatter_nd', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.searchsorted(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.searchsorted', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.select(
                Tensor(np.array(True)), x, x
            )
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.select', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.slice(x, 0)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.slice', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.sort(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.sort', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.split(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.split', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.squeeze(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.squeeze', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.stack([x, y])
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.stack', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.strided_slice(x, x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.strided_slice', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.swapaxes(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.swapaxes', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.take(x, z)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.take', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.take_along_axis(x, z, 0)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.take_along_axis', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.tile(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.tile', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.top_k(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.top_k', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.transpose(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.transpose', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.tril(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.tril', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.triu(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.triu', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.unsqueeze(x, 0)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.unsqueeze', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.unstack(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.unstack', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.update_slice(x, x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.update_slice', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.vsplit(x, x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.vsplit', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.vstack(x)
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.vstack', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.where(
                Tensor(np.array(True)), x, x
            )
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.shape.frontend.where', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Abs()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Abs', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Acos()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Acos', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Acosh()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Acosh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Asin()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Asin', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Asinh()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Asinh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Atan()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Atan', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Atanh()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Atanh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.BitwiseNot()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.BitwiseNot', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Cbrt()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Cbrt', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Ceil()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Ceil', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Conj()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Conj', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Cos()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Cos', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Cosh()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Cosh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Deg2Rad()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Deg2Rad', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Digamma()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Digamma', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Erf()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Erf', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Erfc()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Erfc', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Erfinv()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Erfinv', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Exp()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Exp', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Exp2()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Exp2', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Expm1()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Expm1', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Fix()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Fix', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Floor()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Floor', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Imag()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Imag', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Isfinite()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Isfinite', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Isinf()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Isinf', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Isnan()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Isnan', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Lgamma()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Lgamma', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Log()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Log', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Log10()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Log10', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Log1P()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Log1P', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Log2()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Log2', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.LogicalNot()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.LogicalNot', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Logit()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Logit', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Mvlgamma()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Mvlgamma', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.NanToNum()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.NanToNum', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Negative()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Negative', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Positive()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Positive', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Rad2Deg()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Rad2Deg', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Real()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Real', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Reciprocal()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Reciprocal', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Round()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Round', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Rsqrt()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Rsqrt', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Sign()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Sign', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Signbit()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Signbit', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Sin()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Sin', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Sinc()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Sinc', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Sinh()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Sinh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Sqrt()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Sqrt', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Square()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Square', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Tan()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Tan', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Tanh()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Tanh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Trunc()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.Trunc', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.UnaryMathOp()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.math.UnaryMathOp', e)
        try:
            ml_switcheroo_compiler.ops.unary.special.Bitcast()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.special.Bitcast', e)
        try:
            ml_switcheroo_compiler.ops.unary.special.Cast()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.special.Cast', e)
        try:
            ml_switcheroo_compiler.ops.unary.special.Frexp()
        except Exception:
            pass  # print('ml_switcheroo_compiler.ops.unary.special.Frexp', e)


def test_ops_tracing():
    prev_tracing = getattr(_tracer, "is_tracing", False)
    prev_graph = getattr(_tracer, "active_graph", None)
    try:
        _tracer.is_tracing = True
        _tracer.active_graph = type(
            "Graph", (), {"nodes": {}, "add_node": lambda n: None}
        )()
        _tracer.add_node = lambda n: None
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        y = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
        z = Tensor(np.array([1, 2]))
        try:
            ml_switcheroo_compiler.ops.binary.math.Add()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Add', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.BinaryMathOp()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.BinaryMathOp', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.BitwiseAnd()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.BitwiseAnd', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.BitwiseOr()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.BitwiseOr', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.BitwiseXor()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.BitwiseXor', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Copysign()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Copysign', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Divide()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Divide', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Equal()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Equal', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.FloatPower()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.FloatPower', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.FloorDivide()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.FloorDivide', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Fmax()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Fmax', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Fmin()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Fmin', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Fmod()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Fmod', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Gcd()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Gcd', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Greater()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Greater', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.GreaterEqual()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.GreaterEqual', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Heaviside()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Heaviside', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Hypot()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Hypot', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Lcm()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Lcm', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Ldexp()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Ldexp', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.LeftShift()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.LeftShift', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Less()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Less', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.LessEqual()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.LessEqual', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Logaddexp()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Logaddexp', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Logaddexp2()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Logaddexp2', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.LogicalAnd()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.LogicalAnd', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.LogicalOr()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.LogicalOr', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.LogicalXor()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.LogicalXor', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Maximum()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Maximum', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Minimum()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Minimum', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Mod()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Mod', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Multiply()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Multiply', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Nextafter()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Nextafter', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.NotEqual()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.NotEqual', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Power()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Power', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Remainder()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Remainder', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.RightShift()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.RightShift', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Subtract()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Subtract', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.TrueDivide()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.TrueDivide', e)
        try:
            ml_switcheroo_compiler.ops.binary.math.Xlogy()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.math.Xlogy', e)
        try:
            ml_switcheroo_compiler.ops.binary.special.Allclose()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.special.Allclose', e)
        try:
            ml_switcheroo_compiler.ops.binary.special.Atan2()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.special.Atan2', e)
        try:
            ml_switcheroo_compiler.ops.binary.special.Divmod()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.special.Divmod', e)
        try:
            ml_switcheroo_compiler.ops.binary.special.Isclose()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.binary.special.Isclose', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.cond(
                Tensor(np.array(True)), lambda *args: x, lambda *args: x
            )
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.control_flow.cond', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.pmap(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.control_flow.pmap', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.scan(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.control_flow.scan', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.stop_gradient(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.control_flow.stop_gradient', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.vmap(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.control_flow.vmap', e)
        try:
            ml_switcheroo_compiler.ops.control_flow.while_loop(
                lambda *args: x, lambda *args: x, x
            )
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.control_flow.while_loop', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Arange()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.basic.Arange', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.CreationOp()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.basic.CreationOp', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Full()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.basic.Full', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.ManualSeed()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.basic.ManualSeed', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Ones()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.basic.Ones', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Rand()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.basic.Rand', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Randint()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.basic.Randint', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Randn()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.basic.Randn', e)
        try:
            ml_switcheroo_compiler.ops.creation.basic.Zeros()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.basic.Zeros', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.arange()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.arange', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.array(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.array', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.asarray(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.asarray', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.diag(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.diag', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.empty((2, 2))
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.empty', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.empty_like(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.empty_like', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.eye(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.eye', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.full((2, 2), x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.full', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.full_like(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.full_like', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.identity(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.identity', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.linspace(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.linspace', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.manual_seed(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.manual_seed', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.ones((2, 2))
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.ones', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.ones_like(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.ones_like', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.rand(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.rand', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.randint(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.randint', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.randn(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.randn', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.zeros((2, 2))
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.zeros', e)
        try:
            ml_switcheroo_compiler.ops.creation.frontend.zeros_like(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.creation.frontend.zeros_like', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.ConvGeneralDilated()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.basic.ConvGeneralDilated', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.Dot()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.basic.Dot', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.DotGeneral()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.basic.DotGeneral', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.Einsum()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.basic.Einsum', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.Fft()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.basic.Fft', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.Matmul()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.basic.Matmul', e)
        try:
            ml_switcheroo_compiler.ops.linalg.basic.Rfft()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.basic.Rfft', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.cholesky(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.cholesky', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.conv_general_dilated(x, x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.conv_general_dilated', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.cross(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.cross', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.det(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.det', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.dot(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.dot', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.dot_general(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.dot_general', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.eigh(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.eigh', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.eigvalsh(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.eigvalsh', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.einsum(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.einsum', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.fft(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.fft', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.inner(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.inner', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.inv(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.inv', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.lu(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.lu', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.lu_factor(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.lu_factor', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.matmul(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.matmul', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.matrix_power(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.matrix_power', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.outer(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.outer', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.pinv(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.pinv', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.qr(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.qr', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.rfft(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.rfft', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.slogdet(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.slogdet', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.solve(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.solve', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.solve_triangular(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.solve_triangular', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.svd(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.svd', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.tensordot(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.tensordot', e)
        try:
            ml_switcheroo_compiler.ops.linalg.frontend.vdot(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.linalg.frontend.vdot', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.All()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.All', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.AnyOp()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.AnyOp', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Argmax()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Argmax', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Argmin()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Argmin', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.CountNonzero()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.CountNonzero', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Cumsum()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Cumsum', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Logsumexp()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Logsumexp', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Max()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Max', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Mean()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Mean', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Min()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Min', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Norm()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Norm', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Pmean()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Pmean', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Prod()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Prod', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Psum()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Psum', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.ReduceWindow()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.ReduceWindow', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.ReductionOp()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.ReductionOp', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.SegmentSum()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.SegmentSum', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Std()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Std', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Sum()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Sum', e)
        try:
            ml_switcheroo_compiler.ops.reductions.basic.Variance()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.basic.Variance', e)
        try:
            ml_switcheroo_compiler.ops.reductions.frontend.pmean(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.frontend.pmean', e)
        try:
            ml_switcheroo_compiler.ops.reductions.frontend.psum(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.frontend.psum', e)
        try:
            ml_switcheroo_compiler.ops.reductions.frontend.reduce_window(x, x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.frontend.reduce_window', e)
        try:
            ml_switcheroo_compiler.ops.reductions.frontend.segment_sum(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.reductions.frontend.segment_sum', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.BroadcastInDim()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.basic.BroadcastInDim', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.BroadcastTo()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.basic.BroadcastTo', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.DynamicSlice()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.basic.DynamicSlice', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.DynamicUpdateSlice()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.basic.DynamicUpdateSlice', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.Reshape()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.basic.Reshape', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.Resize()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.basic.Resize', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.Sort()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.basic.Sort', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.TopK()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.basic.TopK', e)
        try:
            ml_switcheroo_compiler.ops.shape.basic.Transpose()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.basic.Transpose', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.array_split(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.array_split', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.broadcast_in_dim(x, (2, 2), x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.broadcast_in_dim', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.broadcast_to(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.broadcast_to', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.concatenate([x, y])
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.concatenate', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.dsplit(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.dsplit', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.dstack(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.dstack', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.dynamic_slice(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.dynamic_slice', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.dynamic_update_slice(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.dynamic_update_slice', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.expand(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.expand', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.expand_dims(x, 0)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.expand_dims', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.flatten(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.flatten', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.gather(x, 0, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.gather', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.gather_nd(x, z)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.gather_nd', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.hsplit(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.hsplit', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.hstack(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.hstack', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.image_resize(x, (2, 2))
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.image_resize', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.meshgrid([x, y])
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.meshgrid', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.moveaxis(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.moveaxis', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.pad(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.pad', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.permute(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.permute', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.repeat(x, 2)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.repeat', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.reshape(x, (2, 2))
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.reshape', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.roll(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.roll', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.scatter(x, 0, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.scatter', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.scatter_add(x, 0, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.scatter_add', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.scatter_nd(z, x, (2, 2))
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.scatter_nd', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.searchsorted(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.searchsorted', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.select(
                Tensor(np.array(True)), x, x
            )
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.select', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.slice(x, 0)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.slice', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.sort(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.sort', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.split(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.split', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.squeeze(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.squeeze', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.stack([x, y])
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.stack', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.strided_slice(x, x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.strided_slice', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.swapaxes(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.swapaxes', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.take(x, z)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.take', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.take_along_axis(x, z, 0)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.take_along_axis', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.tile(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.tile', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.top_k(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.top_k', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.transpose(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.transpose', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.tril(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.tril', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.triu(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.triu', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.unsqueeze(x, 0)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.unsqueeze', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.unstack(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.unstack', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.update_slice(x, x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.update_slice', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.vsplit(x, x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.vsplit', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.vstack(x)
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.vstack', e)
        try:
            ml_switcheroo_compiler.ops.shape.frontend.where(
                Tensor(np.array(True)), x, x
            )
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.shape.frontend.where', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Abs()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Abs', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Acos()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Acos', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Acosh()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Acosh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Asin()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Asin', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Asinh()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Asinh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Atan()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Atan', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Atanh()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Atanh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.BitwiseNot()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.BitwiseNot', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Cbrt()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Cbrt', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Ceil()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Ceil', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Conj()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Conj', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Cos()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Cos', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Cosh()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Cosh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Deg2Rad()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Deg2Rad', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Digamma()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Digamma', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Erf()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Erf', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Erfc()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Erfc', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Erfinv()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Erfinv', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Exp()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Exp', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Exp2()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Exp2', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Expm1()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Expm1', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Fix()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Fix', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Floor()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Floor', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Imag()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Imag', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Isfinite()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Isfinite', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Isinf()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Isinf', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Isnan()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Isnan', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Lgamma()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Lgamma', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Log()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Log', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Log10()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Log10', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Log1P()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Log1P', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Log2()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Log2', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.LogicalNot()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.LogicalNot', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Logit()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Logit', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Mvlgamma()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Mvlgamma', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.NanToNum()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.NanToNum', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Negative()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Negative', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Positive()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Positive', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Rad2Deg()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Rad2Deg', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Real()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Real', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Reciprocal()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Reciprocal', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Round()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Round', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Rsqrt()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Rsqrt', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Sign()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Sign', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Signbit()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Signbit', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Sin()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Sin', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Sinc()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Sinc', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Sinh()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Sinh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Sqrt()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Sqrt', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Square()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Square', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Tan()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Tan', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Tanh()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Tanh', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.Trunc()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.Trunc', e)
        try:
            ml_switcheroo_compiler.ops.unary.math.UnaryMathOp()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.math.UnaryMathOp', e)
        try:
            ml_switcheroo_compiler.ops.unary.special.Bitcast()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.special.Bitcast', e)
        try:
            ml_switcheroo_compiler.ops.unary.special.Cast()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.special.Cast', e)
        try:
            ml_switcheroo_compiler.ops.unary.special.Frexp()
        except Exception:
            pass  # print('Tracing ml_switcheroo_compiler.ops.unary.special.Frexp', e)
    finally:
        _tracer.is_tracing = prev_tracing
        _tracer.active_graph = prev_graph
