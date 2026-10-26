"""Unit tests for `test_coverage_all`."""

from __future__ import annotations

import math

import zero_tensorflow as tf
from zero_tensorflow import Tensor, debugging
from zero_tensorflow.raw_ops import _extract_arg

import pytest


def test_math_coverage() -> None:
    """Verify `tf.math` operations produce accurate numerical `Tensor` outputs."""
    x = Tensor([1.0, 4.0])
    y = Tensor([2.0, 2.0])

    debugging.assert_near(tf.math.add(x, y), [3.0, 6.0])
    debugging.assert_near(tf.math.subtract(x, y), [-1.0, 2.0])
    debugging.assert_near(tf.math.multiply(x, y), [2.0, 8.0])
    debugging.assert_near(tf.math.divide(x, y), [0.5, 2.0])
    debugging.assert_near(tf.math.sqrt(x), [1.0, 2.0])
    debugging.assert_near(tf.math.square(x), [1.0, 16.0])
    debugging.assert_near(tf.math.abs(Tensor([-3.0, 4.0])), [3.0, 4.0])
    debugging.assert_near(tf.math.reduce_sum(x), 5.0)
    debugging.assert_near(tf.math.reduce_mean(x), 2.5)
    debugging.assert_near(tf.math.reduce_max(x), 4.0)
    debugging.assert_near(tf.math.reduce_min(x), 1.0)
    debugging.assert_near(tf.math.reduce_prod(x), 4.0)
    debugging.assert_near(tf.math.maximum(x, y), [2.0, 4.0])
    debugging.assert_near(tf.math.minimum(x, y), [1.0, 2.0])
    debugging.assert_near(tf.math.pow(x, y), [1.0, 16.0])


def test_nn_coverage() -> None:
    """Verify `tf.nn` activation and normalization operations produce accurate outputs."""
    x = Tensor([-1.0, 0.0, 2.0])
    debugging.assert_near(tf.nn.relu(x), [0.0, 0.0, 2.0])
    debugging.assert_near(tf.nn.leaky_relu(Tensor([0.0, 2.0])), [0.0, 2.0])
    debugging.assert_near(tf.nn.sigmoid(Tensor([0.0])), [0.5])
    debugging.assert_near(tf.nn.tanh(Tensor([0.0])), [0.0])
    debugging.assert_near(tf.nn.softmax(Tensor([[1.0, 1.0]])), [[0.5, 0.5]])


def test_linalg_coverage() -> None:
    """Verify `tf.linalg` matrix operations produce accurate `Tensor` outputs."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[2.0, 0.0], [0.0, 2.0]])
    debugging.assert_near(tf.linalg.matmul(a, b), [[2.0, 4.0], [6.0, 8.0]])
    debugging.assert_near(tf.linalg.det(b), 4.0)
    debugging.assert_near(tf.linalg.cholesky(b), [[math.sqrt(2.0), 0.0], [0.0, math.sqrt(2.0)]])
    debugging.assert_near(tf.linalg.inv(b), [[0.5, 0.0], [0.0, 0.5]])
    debugging.assert_near(tf.linalg.diag(Tensor([2.0, 3.0])), [[2.0, 0.0], [0.0, 3.0]])


def test_bitwise_coverage() -> None:
    """Verify `tf.bitwise` integer bitwise operations produce accurate outputs."""
    a = Tensor([5, 3], dtype=tf.int32)
    b = Tensor([3, 6], dtype=tf.int32)
    debugging.assert_equal(tf.bitwise.bitwise_and(a, b), [1, 2])
    debugging.assert_equal(tf.bitwise.bitwise_or(a, b), [7, 7])
    debugging.assert_equal(tf.bitwise.bitwise_xor(a, b), [6, 5])
    debugging.assert_equal(
        tf.bitwise.left_shift(Tensor([1, 2], dtype=tf.int32), Tensor([2, 1], dtype=tf.int32)),
        [4, 4],
    )
    debugging.assert_equal(
        tf.bitwise.right_shift(Tensor([8, 4], dtype=tf.int32), Tensor([2, 1], dtype=tf.int32)),
        [2, 2],
    )
    debugging.assert_equal(tf.bitwise.invert(Tensor([0, 1, -1], dtype=tf.int8)), [-1, -2, 0])
    debugging.assert_equal(tf.bitwise.invert(Tensor([0, 1, 255], dtype=tf.uint8)), [255, 254, 0])
    debugging.assert_equal(tf.bitwise.invert(Tensor([0, 1, -1], dtype=tf.int16)), [-1, -2, 0])
    debugging.assert_equal(
        tf.bitwise.invert(Tensor([0, 1, 65535], dtype=tf.uint16)), [65535, 65534, 0]
    )
    debugging.assert_equal(tf.bitwise.invert(Tensor([0, 1, -1], dtype=tf.int32)), [-1, -2, 0])
    debugging.assert_equal(
        tf.bitwise.invert(Tensor([0, 1, 4294967295], dtype=tf.uint32)),
        [4294967295, 4294967294, 0],
    )
    debugging.assert_equal(tf.bitwise.invert(Tensor([0, 1, -1], dtype=tf.int64)), [-1, -2, 0])
    debugging.assert_equal(
        tf.bitwise.invert(Tensor([0, 1, 18446744073709551615], dtype=tf.uint64)),
        [18446744073709551615, 18446744073709551614, 0],
    )


def test_toplevel_coverage() -> None:
    """Verify top-level TensorFlow operations (`zeros`, `ones`, `eye`, `fill`, `range`, `linspace`, `where`)."""
    debugging.assert_near(tf.zeros((2, 2)), [[0.0, 0.0], [0.0, 0.0]])
    debugging.assert_near(tf.ones((2, 2)), [[1.0, 1.0], [1.0, 1.0]])
    debugging.assert_near(tf.eye(2), [[1.0, 0.0], [0.0, 1.0]])
    debugging.assert_near(tf.fill((2, 2), 7.0), [[7.0, 7.0], [7.0, 7.0]])
    debugging.assert_near(tf.range(0, 4, 1), [0, 1, 2, 3])
    debugging.assert_near(tf.linspace(0.0, 1.0, 3), [0.0, 0.5, 1.0])
    cond = Tensor([True, False])
    debugging.assert_near(
        tf.where(cond, Tensor([10.0, 20.0]), Tensor([30.0, 40.0])),
        [10.0, 40.0],
    )


def test_gradient_elementary_functions() -> None:
    """Verify `GradientTape` differentiates polynomials, linear expressions, and transcendental functions."""
    x = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x * x
    grad_sq = tape.gradient(y, x)
    assert isinstance(grad_sq, Tensor)
    debugging.assert_near(grad_sq, 6.0)

    with tf.GradientTape() as tape2:
        tape2.watch(x)
        y2 = x * 3.0 + tf.Tensor(2.0)
    grad_lin = tape2.gradient(y2, x)
    assert isinstance(grad_lin, Tensor)
    debugging.assert_near(grad_lin, 3.0)

    with tf.GradientTape() as tape3:
        tape3.watch(x)
        y3 = tf.math.sin(x)
    grad_sin = tape3.gradient(y3, x)
    assert isinstance(grad_sin, Tensor)
    debugging.assert_near(grad_sin, math.cos(3.0))


def test_gradient_val_2() -> None:
    """Verify `GradientTape` differentiates `x * 2.0` to `2.0`."""
    x = tf.Variable(2.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x * 2.0
    grad = tape.gradient(y, x)
    assert isinstance(grad, Tensor) and grad.numpy() == 2.0


def test_gradient_val_1() -> None:
    """Verify `GradientTape` differentiates `x + 1.0` to `1.0`."""
    x = tf.Variable(2.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x + 1.0
    grad = tape.gradient(y, x)
    assert isinstance(grad, Tensor) and grad.numpy() == 1.0


def test_raw_ops_routing() -> None:
    """Verify routed `tf.raw_ops` methods execute underlying tensor operations."""
    x = Tensor([1.0, 4.0])
    y = Tensor([2.0, 2.0])

    res_abs = tf.raw_ops.Abs(x=Tensor([-2.0, 3.0]))
    assert isinstance(res_abs, Tensor)
    debugging.assert_near(res_abs, [2.0, 3.0])

    res_add = tf.raw_ops.Add(x, y)
    assert isinstance(res_add, Tensor)
    debugging.assert_near(res_add, [3.0, 6.0])

    res_add_v2 = tf.raw_ops.AddV2(x=x, y=y)
    assert isinstance(res_add_v2, Tensor)
    debugging.assert_near(res_add_v2, [3.0, 6.0])

    a_mat = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b_mat = Tensor([[2.0, 0.0], [0.0, 2.0]])
    res_mm = tf.raw_ops.MatMul(a=a_mat, b=b_mat)
    assert isinstance(res_mm, Tensor)
    debugging.assert_near(res_mm, [[2.0, 4.0], [6.0, 8.0]])

    res_bmm = tf.raw_ops.BatchMatMulV2(a_mat, b_mat)
    assert isinstance(res_bmm, Tensor)
    debugging.assert_near(res_bmm, [[2.0, 4.0], [6.0, 8.0]])

    res_relu = tf.raw_ops.Relu(features=Tensor([-1.0, 2.0]))
    assert isinstance(res_relu, Tensor)
    debugging.assert_near(res_relu, [0.0, 2.0])

    res_sig = tf.raw_ops.Sigmoid(Tensor([0.0]))
    assert isinstance(res_sig, Tensor)
    debugging.assert_near(res_sig, [0.5])

    res_tanh = tf.raw_ops.Tanh(Tensor([0.0]))
    assert isinstance(res_tanh, Tensor)
    debugging.assert_near(res_tanh, [0.0])

    res_sm = tf.raw_ops.Softmax(logits=Tensor([[1.0, 1.0]]))
    assert isinstance(res_sm, Tensor)
    debugging.assert_near(res_sm, [[0.5, 0.5]])

    res_sub = tf.raw_ops.Sub(x, y)
    assert isinstance(res_sub, Tensor)
    debugging.assert_near(res_sub, [-1.0, 2.0])

    res_mul = tf.raw_ops.Mul(x, y)
    assert isinstance(res_mul, Tensor)
    debugging.assert_near(res_mul, [2.0, 8.0])

    res_div = tf.raw_ops.RealDiv(x, y)
    assert isinstance(res_div, Tensor)
    debugging.assert_near(res_div, [0.5, 2.0])

    res_sqrt = tf.raw_ops.Sqrt(x)
    assert isinstance(res_sqrt, Tensor)
    debugging.assert_near(res_sqrt, [1.0, 2.0])

    res_exp = tf.raw_ops.Exp(Tensor([0.0]))
    assert isinstance(res_exp, Tensor)
    debugging.assert_near(res_exp, [1.0])

    res_log = tf.raw_ops.Log(Tensor([1.0]))
    assert isinstance(res_log, Tensor)
    debugging.assert_near(res_log, [0.0])

    res_sin = tf.raw_ops.Sin(Tensor([0.0]))
    assert isinstance(res_sin, Tensor)
    debugging.assert_near(res_sin, [0.0])

    res_cos = tf.raw_ops.Cos(Tensor([0.0]))
    assert isinstance(res_cos, Tensor)
    debugging.assert_near(res_cos, [1.0])

    res_reshape = tf.raw_ops.Reshape(tensor=Tensor([1.0, 2.0, 3.0, 4.0]), shape=(2, 2))
    assert isinstance(res_reshape, Tensor) and res_reshape.shape == (2, 2)

    res_tr1 = tf.raw_ops.Transpose(a_mat)
    assert isinstance(res_tr1, Tensor)
    debugging.assert_near(res_tr1, [[1.0, 3.0], [2.0, 4.0]])

    res_tr2 = tf.raw_ops.Transpose(a_mat, (1, 0))
    assert isinstance(res_tr2, Tensor)
    debugging.assert_near(res_tr2, [[1.0, 3.0], [2.0, 4.0]])

    res_concat = tf.raw_ops.ConcatV2([x, y], 0)
    assert isinstance(res_concat, Tensor)
    debugging.assert_near(res_concat, [1.0, 4.0, 2.0, 2.0])

    res_pack = tf.raw_ops.Pack([x, y], axis=0)
    assert isinstance(res_pack, Tensor) and res_pack.shape == (2, 2)

    res_unpack = tf.raw_ops.Unpack(res_pack, axis=0)
    assert isinstance(res_unpack, list) and len(res_unpack) == 2

    res_slice = tf.raw_ops.Slice(Tensor([10, 20, 30, 40]), [1], [2])
    assert isinstance(res_slice, Tensor)
    debugging.assert_equal(res_slice, [20, 30])

    res_sslice = tf.raw_ops.StridedSlice(
        input=Tensor([10, 20, 30, 40]), begin=[0], end=[4], strides=[2]
    )
    assert isinstance(res_sslice, Tensor)
    debugging.assert_equal(res_sslice, [10, 30])

    res_gather = tf.raw_ops.GatherV2(params=Tensor([10, 20, 30]), indices=Tensor([2, 0]))
    assert isinstance(res_gather, Tensor)
    debugging.assert_equal(res_gather, [30, 10])

    res_gnd = tf.raw_ops.GatherNd(Tensor([10, 20, 30]), Tensor([1]))
    assert isinstance(res_gnd, Tensor)

    res_snd = tf.raw_ops.ScatterNd(indices=[0, 2], updates=[9.0, 8.0], shape=(3,))
    assert isinstance(res_snd, Tensor)

    res_where = tf.raw_ops.Where(Tensor([True, False]), x, y)
    assert isinstance(res_where, Tensor)
    debugging.assert_near(res_where, [1.0, 2.0])

    res_cast = tf.raw_ops.Cast(x=Tensor([1.5, 2.5]), DstT=tf.int32)
    assert isinstance(res_cast, Tensor)
    debugging.assert_equal(res_cast, [1, 2])

    assert isinstance(tf.raw_ops.Equal(x, y), Tensor)
    assert isinstance(tf.raw_ops.NotEqual(x, y), Tensor)
    assert isinstance(tf.raw_ops.Greater(x, y), Tensor)
    assert isinstance(tf.raw_ops.Less(x, y), Tensor)
    assert isinstance(tf.raw_ops.LogicalAnd(Tensor([True]), Tensor([False])), Tensor)
    assert isinstance(tf.raw_ops.LogicalOr(Tensor([True]), Tensor([False])), Tensor)
    assert isinstance(tf.raw_ops.LogicalNot(Tensor([True])), Tensor)

    bi_a = Tensor([5], dtype=tf.int32)
    bi_b = Tensor([3], dtype=tf.int32)
    assert isinstance(tf.raw_ops.BitwiseAnd(bi_a, bi_b), Tensor)
    assert isinstance(tf.raw_ops.BitwiseOr(bi_a, bi_b), Tensor)
    assert isinstance(tf.raw_ops.BitwiseXor(bi_a, bi_b), Tensor)
    assert isinstance(tf.raw_ops.LeftShift(bi_a, bi_b), Tensor)
    assert isinstance(tf.raw_ops.RightShift(bi_a, bi_b), Tensor)

    assert isinstance(tf.raw_ops.Cholesky(b_mat), Tensor)
    assert isinstance(tf.raw_ops.MatrixInverse(b_mat), Tensor)
    assert isinstance(tf.raw_ops.Qr(b_mat), (Tensor, tuple, list))
    assert isinstance(tf.raw_ops.Svd(b_mat), (Tensor, tuple, list))
    assert tf.raw_ops.StringLower("HELLO") == "hello"
    assert tf.raw_ops.StringUpper("hello") == "HELLO"
    assert tf.raw_ops.StringStrip("  hi  ") == "hi"
    assert tf.raw_ops.StringLength("hello") == 5

    # Test new Tensor & Variable methods
    t_base = Tensor([1.0, 2.0])
    assert t_base.experimental_ref().deref() is t_base
    assert t_base.get_shape() == (2,)
    t_base.set_shape((2, 1))
    assert t_base.get_shape() == (2, 1)
    t_base.set_shape((None, 2))
    assert t_base.get_shape() == (0, 2)
    t_base.set_shape(None)
    assert t_base.eval() is not None
    assert t_base() is t_base

    v_cnt = tf.Variable(0, dtype=tf.int32)
    assert v_cnt.initialized_value().numpy() == 0
    assert v_cnt.value() is v_cnt
    assert tf.Variable.value() is not None
    assert v_cnt.count_up_to(2).numpy() == 0
    assert v_cnt.count_up_to(2).numpy() == 1
    with pytest.raises(IndexError):
        v_cnt.count_up_to(2)
    v_cnt.load(5)
    assert v_cnt.numpy() == 5
    v_sc = tf.Variable(4, dtype=tf.int32)
    assert v_sc.count_up_to(5).numpy() == 4
    v_vec = tf.Variable([10.0, 20.0])
    debugging.assert_near(v_vec.gather_nd([1]), [20.0])
    assert v_vec.to_proto(export_scope="s")["export_scope"] == "s"
    assert isinstance(tf.Variable.from_proto("proto", import_scope="scope"), tf.Variable)
    assert isinstance(tf.Variable.from_proto(3.0), tf.Variable)
    v_vec.scatter_update([2.0, 4.0])
    v_vec.scatter_add([1.0, 1.0])
    v_vec.scatter_sub([1.0, 1.0])
    v_vec.scatter_mul([2.0, 2.0])
    v_vec.scatter_div([2.0, 2.0])
    v_vec.scatter_max([5.0, 1.0])
    v_vec.scatter_min([3.0, 10.0])
    debugging.assert_near(v_vec, [3.0, 4.0])

    # Test new array_ops and math_ops methods
    assert tf.check_pinned(None) is None
    debugging.assert_near(tf.check_pinned(Tensor([1.0])), [1.0])
    assert tf.irfftnd(None) is None
    rfft_out = tf.rfftnd(Tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert isinstance(tf.irfftnd(rfft_out), Tensor)
    debugging.assert_near(tf.scalar_mul(3.0, Tensor([1.0, 2.0])), [3.0, 6.0])
    debugging.assert_near(tf.truediv(Tensor([4.0, 6.0]), Tensor([2.0, 3.0])), [2.0, 2.0])
    assert isinstance(tf.scan(lambda a, x: a + x, Tensor([1.0, 2.0, 3.0])), Tensor)

    assert isinstance(tf.math.bincount(Tensor([0, 1, 1], dtype=tf.int32)), Tensor)
    assert isinstance(tf.math.sobol_sample(2, 4), Tensor)
    assert isinstance(tf.repeat(Tensor([1.0, 2.0]), 2), Tensor)
    assert isinstance(tf.roll(Tensor([1.0, 2.0, 3.0]), 1, 0), Tensor)
    assert isinstance(tf.unstack(Tensor([[1.0, 2.0], [3.0, 4.0]])), (Tensor, list, tuple))

    # Exercise newly added v0.0.5 raw_ops endpoints and _extract_arg kwargs fallback
    assert _extract_arg((), {"x": 42}, 0, "x") == 42
    assert tf.raw_ops.CheckPinned() is None
    assert tf.raw_ops.ConvertToListOfSparseCoreCooTensors() is None
    assert tf.raw_ops.ConvertToSparseCoreCsrWrappedCooTensor() is None
    assert tf.raw_ops.DecodeJxl() is None
    assert tf.raw_ops.DecodeWebP() is None
    assert tf.raw_ops.GetStatsFromListOfSparseCoreCooTensors() is None
    assert tf.raw_ops.GlobalShuffleDataset() is None
    assert tf.raw_ops.IndexFlatMapDataset() is None
    assert tf.raw_ops.IteratorGetModelProto() is None
    assert tf.raw_ops.SortListOfSparseCoreCooTensors() is None
    assert tf.raw_ops.TPUDummyInput() is None
    assert tf.raw_ops.WeightedFlatMapDataset() is None
    assert tf.raw_ops.XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdagradAndCsrInput() is None
    assert (
        tf.raw_ops.XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdagradMomentumAndCsrInput()
        is None
    )
    assert tf.raw_ops.XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdamAndCsrInput() is None
    assert tf.raw_ops.XlaSparseDenseMatmulCustomCombinerOnTcGradWithCsrInput() is None
    assert tf.raw_ops.XlaSparseDenseMatmulCustomCombinerOnTcGradWithFtrlAndCsrInput() is None
    assert tf.raw_ops.XlaSparseDenseMatmulCustomCombinerOnTcGradWithSgdAndCsrInput() is None
    assert tf.raw_ops.XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInput() is None
    assert tf.raw_ops.XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize() is None
    assert tf.raw_ops.XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize() is None
    assert tf.raw_ops.XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize() is None
    assert tf.raw_ops.XlaSparseDenseMatmulGradWithCsrInput() is None
    assert tf.raw_ops.XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize() is None
    assert tf.raw_ops.XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize() is None
    assert tf.raw_ops.XlaSparseDenseMatmulWithStaticBufferSize() is None
