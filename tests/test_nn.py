"""Unit tests for `test_nn`."""

from __future__ import annotations

import math

import zero_tensorflow as tf
from zero_tensorflow import Tensor, debugging, function, nn
from zero_tensorflow._typing import TensorOperand

import pytest


def test_elu() -> None:
    """Execute `test_elu`."""
    x = Tensor([-1.0, 0.0, 1.0], dtype=tf.float32)
    y = nn.elu(x)
    debugging.assert_near(y, [math.exp(-1.0) - 1.0, 0.0, 1.0], rtol=1e-05)
    with pytest.raises(ValueError):
        nn.elu(None)


def test_leaky_relu() -> None:
    """Execute `test_leaky_relu`."""
    x = Tensor([-1.0, 0.0, 1.0], dtype=tf.float32)
    y = nn.leaky_relu(x, alpha=0.1)
    debugging.assert_near(y, [-0.1, 0.0, 1.0], rtol=1e-05)
    with pytest.raises(ValueError):
        nn.leaky_relu(None)


def test_relu() -> None:
    """Execute `test_relu`."""
    x = Tensor([-1.0, 0.0, 1.0], dtype=tf.float32)
    y = nn.relu(x)
    debugging.assert_near(y, [0.0, 0.0, 1.0], rtol=1e-05)
    with pytest.raises(ValueError):
        nn.relu(None)


def test_selu() -> None:
    """Execute `test_selu`."""
    x = Tensor([-1.0, 0.0, 1.0], dtype=tf.float32)
    y = nn.selu(x)
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    expected_neg = scale * alpha * (math.exp(-1.0) - 1.0)
    expected = [expected_neg, 0.0, scale * 1.0]
    debugging.assert_near(y, expected, rtol=1e-05)
    with pytest.raises(ValueError):
        nn.selu(None)


def test_sigmoid() -> None:
    """Execute `test_sigmoid`."""
    x = Tensor([-1.0, 0.0, 1.0], dtype=tf.float32)
    y = nn.sigmoid(x)
    expected = [1.0 / (1.0 + math.exp(-v)) for v in (-1.0, 0.0, 1.0)]
    debugging.assert_near(y, expected, rtol=1e-05)
    with pytest.raises(ValueError):
        nn.sigmoid(None)


def test_softmax() -> None:
    """Execute `test_softmax`."""
    x = Tensor([[1.0, 2.0, 3.0]], dtype=tf.float32)
    y = nn.softmax(x)
    e_vals = [math.exp(v - 3.0) for v in (1.0, 2.0, 3.0)]
    s_val = sum(e_vals)
    expected = [[v / s_val for v in e_vals]]
    debugging.assert_near(y, expected, rtol=1e-05)
    y_default = nn.softmax(x, axis=None)
    debugging.assert_near(y_default, expected, rtol=1e-05)
    with pytest.raises(ValueError):
        nn.softmax(None)


def test_tanh() -> None:
    """Execute `test_tanh`."""
    x = Tensor([-1.0, 0.0, 1.0], dtype=tf.float32)
    y = nn.tanh(x)
    debugging.assert_near(y, [math.tanh(-1.0), math.tanh(0.0), math.tanh(1.0)], rtol=1e-05)
    with pytest.raises(ValueError):
        nn.tanh(None)


def test_nn_traced() -> None:
    """Execute `test_nn_traced`."""

    @function
    def nn_f(x: Tensor) -> tuple[Tensor, ...]:
        """Execute `nn_f`.

        Args:
            x: Parameter `x`.

        Returns:
            tuple[Tensor, ...]: Computed result.
        """
        return (
            nn.relu(x),
            nn.elu(x),
            nn.leaky_relu(x),
            nn.selu(x),
            nn.sigmoid(x),
            nn.softmax(x),
            nn.tanh(x),
        )

    x = Tensor([-1.0, 0.0, 1.0])
    res = nn_f(x)
    assert all(isinstance(r, Tensor) for r in res)


def test_nn_conv_pool_norm_and_losses() -> None:
    """Verify `nn` convolutions, pooling, batch/LRN normalization, and loss functions."""
    x3d = tf.ones((1, 2, 2, 2, 1), dtype=tf.float32)
    assert isinstance(nn.avg_pool3d(x3d, (1, 1, 1), (1, 1, 1), "VALID"), Tensor)
    assert isinstance(nn.max_pool3d(x3d, (1, 1, 1), (1, 1, 1), "VALID"), Tensor)

    x1d = tf.ones((1, 2, 4), dtype=tf.float32)
    k1d = tf.ones((3, 2, 2), dtype=tf.float32)
    assert isinstance(nn.conv1d(x1d, k1d), Tensor)
    x1d_ncw = tf.ones((1, 2, 4), dtype=tf.float32)
    assert nn.conv1d(
        x1d_ncw, k1d, stride=[1, 1, 1], dilations=[1, 1, 1], data_format="NCW"
    ).shape == (
        1,
        2,
        2,
    )

    x2d = tf.ones((1, 2, 2, 2), dtype=tf.float32)
    k2d = tf.ones((2, 2, 2, 2), dtype=tf.float32)
    assert isinstance(nn.conv2d_transpose(x2d, k2d, 1, "VALID"), Tensor)
    with pytest.raises(NotImplementedError):
        nn.conv2d_transpose()

    x3d_c = tf.ones((1, 1, 2, 2, 2), dtype=tf.float32)
    k3d = tf.ones((1, 1, 1, 1, 1), dtype=tf.float32)
    assert isinstance(nn.conv3d_transpose(x3d_c, k3d, 1, "VALID"), Tensor)
    with pytest.raises(NotImplementedError):
        nn.conv3d_transpose()

    x2d_hwc = tf.ones((1, 4, 4, 2), dtype=tf.float32)
    dw_k = tf.ones((2, 2, 2, 1), dtype=tf.float32)
    assert isinstance(nn.depthwise_conv2d(x2d_hwc, dw_k), Tensor)
    assert isinstance(
        nn.depthwise_conv2d(x2d_hwc, dw_k, strides=(1, 1), dilations=(1, 1)),
        Tensor,
    )
    assert isinstance(
        nn.depthwise_conv2d(x2d_hwc, dw_k, strides=1, dilations=1),
        Tensor,
    )
    x2d_nchw = tf.ones((1, 2, 4, 4), dtype=tf.float32)
    assert nn.depthwise_conv2d(x2d_nchw, dw_k, data_format="NCHW").shape == (1, 2, 3, 3)
    pt_k = tf.ones((1, 1, 2, 3), dtype=tf.float32)
    assert isinstance(nn.separable_conv2d(x2d_hwc, dw_k, pt_k), Tensor)
    assert isinstance(
        nn.separable_conv2d(x2d_hwc, dw_k, pt_k, strides=(1, 1), dilations=(1, 1)),
        Tensor,
    )
    assert isinstance(
        nn.separable_conv2d(x2d_hwc, dw_k, pt_k, strides=1, dilations=1),
        Tensor,
    )
    assert nn.separable_conv2d(x2d_nchw, dw_k, pt_k, data_format="NCHW").shape == (1, 3, 3, 3)

    crelu_out = nn.crelu(Tensor([-1.0, 2.0]))
    debugging.assert_near(crelu_out, [0.0, 2.0, 1.0, 0.0])

    assert isinstance(nn.gelu(Tensor([0.0, 1.0])), Tensor)
    assert isinstance(nn.silu(Tensor([0.0, 1.0])), Tensor)
    debugging.assert_near(nn.silu(Tensor([0.0]), beta=2.0), [0.0])
    assert isinstance(nn.swish(Tensor([0.0, 1.0])), Tensor)
    debugging.assert_near(nn.swish(Tensor([0.0]), beta=2.0), [0.0])

    bn_out = nn.batch_norm_with_global_normalization(
        Tensor([2.0, 4.0]),
        Tensor([1.0, 1.0]),
        Tensor([1.0, 1.0]),
        Tensor([0.0, 0.0]),
        Tensor([2.0, 2.0]),
        variance_epsilon=0.0,
        scale_after_normalization=True,
    )
    debugging.assert_near(bn_out, [2.0, 6.0])

    bn_unscaled = nn.batch_norm_with_global_normalization(
        Tensor([2.0, 4.0]),
        Tensor([1.0, 1.0]),
        Tensor([1.0, 1.0]),
        Tensor([0.0, 0.0]),
        variance_epsilon=0.0,
        scale_after_normalization=False,
    )
    debugging.assert_near(bn_unscaled, [1.0, 3.0])
    with pytest.raises(NotImplementedError):
        nn.batch_norm_with_global_normalization()

    lrn_out = nn.local_response_normalization(
        Tensor([1.0, 2.0, 3.0]), depth_radius=1, bias=1.0, alpha=1.0, beta=1.0
    )
    debugging.assert_near(lrn_out, [1.0 / 6.0, 2.0 / 15.0, 3.0 / 14.0])
    lrn_r0 = nn.local_response_normalization(
        Tensor([0.0, 1.0]), depth_radius=0, bias=1.0, alpha=1.0, beta=1.0
    )
    debugging.assert_near(lrn_r0, [0.0, 0.5])
    with pytest.raises(NotImplementedError):
        nn.local_response_normalization()

    y_true_ctc = Tensor([[1, 2]])
    y_pred_ctc = tf.ones((4, 1, 4), dtype=tf.float32)
    assert isinstance(
        nn.ctc_loss(y_true_ctc, y_pred_ctc, Tensor([2]), Tensor([4])),
        Tensor,
    )

    fmp_in = tf.reshape(tf.cast(tf.range(0, 16, 1), tf.float32), (1, 4, 4, 1))
    fmp_out, row_seq, col_seq = nn.fractional_max_pool(
        fmp_in, pooling_ratio=[1.0, 2.0, 2.0, 1.0], overlapping=True
    )
    assert fmp_out.shape == (1, 2, 2, 1)
    debugging.assert_near(row_seq, [0, 2, 4])
    debugging.assert_near(col_seq, [0, 2, 4])
    fmp_1x1, _, _ = nn.fractional_max_pool(
        tf.ones((1, 2, 2, 1), dtype=tf.float32),
        pooling_ratio=[1.0, 2.0, 2.0, 1.0],
        overlapping=False,
    )
    assert fmp_1x1.shape == (1, 1, 1, 1)
    fmp_short, _, _ = nn.fractional_max_pool(Tensor([1.0, 2.0]), pooling_ratio=[1.0])
    assert fmp_short.shape == (2,)
    with pytest.raises(NotImplementedError):
        nn.fractional_max_pool()

    weights = Tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    biases = Tensor([0.0, 0.0, 0.0])
    labels_2d = Tensor([[0], [1]])
    labels_1d = Tensor([0, 1])
    inputs = Tensor([[1.0, 0.0], [0.0, 1.0]])
    nce_out = nn.nce_loss(weights, biases, labels_2d, inputs, num_sampled=2, num_classes=3)
    assert nce_out.shape == (2,)
    nce_1d = nn.nce_loss(weights, biases, labels_1d, inputs, num_sampled=2, num_classes=3)
    assert nce_1d.shape == (2,)
    ssl_out = nn.sampled_softmax_loss(
        weights, biases, labels_2d, inputs, num_sampled=2, num_classes=3
    )
    assert ssl_out.shape == (2,)
    ssl_1d = nn.sampled_softmax_loss(
        weights, biases, labels_1d, inputs, num_sampled=2, num_classes=3
    )
    assert ssl_1d.shape == (2,)
    with pytest.raises(NotImplementedError):
        nn.nce_loss()
    with pytest.raises(NotImplementedError):
        nn.sampled_softmax_loss()


def test_nn_extended_ops() -> None:
    """Verify additional `nn` pooling, convolution, normalization, and cross-entropy ops."""
    x1 = tf.ones((1, 4, 2), dtype=tf.float32)
    assert nn.avg_pool1d(x1, ksize=2, strides=2).shape == (1, 2, 2)
    assert nn.max_pool1d(x1, ksize=2, strides=2).shape == (1, 2, 2)

    x2 = tf.ones((1, 4, 4, 2), dtype=tf.float32)
    assert nn.avg_pool2d(x2, ksize=[1, 2, 2, 1], strides=[2, 2]).shape == (1, 2, 2, 2)
    assert nn.max_pool2d(x2, ksize=2, strides=2).shape == (1, 2, 2, 2)
    assert nn.avg_pool(x2, ksize=2).shape == (1, 2, 2, 2)
    assert nn.max_pool(x2, ksize=2).shape == (1, 2, 2, 2)

    f2 = tf.ones((2, 2, 2, 3), dtype=tf.float32)
    assert nn.conv2d(x2, f2, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1]).shape == (1, 3, 3, 3)
    assert nn.conv2d(x2, f2, strides=(1, 1), dilations=(1, 1)).shape == (1, 3, 3, 3)
    assert nn.conv2d(x2, f2, strides=1, dilations=None).shape == (1, 3, 3, 3)

    x3 = tf.ones((1, 3, 3, 3, 2), dtype=tf.float32)
    f3 = tf.ones((2, 2, 2, 2, 4), dtype=tf.float32)
    assert nn.conv3d(x3, f3, strides=[1, 1, 1, 1, 1], dilations=[1, 1, 1, 1, 1]).shape == (
        1,
        2,
        2,
        2,
        4,
    )
    assert nn.conv3d(x3, f3, strides=(1, 1, 1), dilations=(1, 1, 1)).shape == (1, 2, 2, 2, 4)
    assert nn.conv3d(x3, f3, strides=1, dilations=None).shape == (1, 2, 2, 2, 4)

    dropped = nn.dropout(Tensor([1.0, 2.0, 3.0]), rate=0.0, noise_shape=[3], seed=42)
    debugging.assert_near(dropped, [1.0, 2.0, 3.0])

    emb_params = Tensor([[3.0, 4.0], [0.0, 2.0]])
    emb_out = nn.embedding_lookup(emb_params, Tensor([0, 1]), max_norm=2.5)
    debugging.assert_near(emb_out, [[1.5, 2.0], [0.0, 2.0]])
    emb_no_clip = nn.embedding_lookup(emb_params, Tensor([0]))
    debugging.assert_near(emb_no_clip, [[3.0, 4.0]])

    m_mean, m_var = nn.moments(Tensor([[1.0, 3.0], [5.0, 7.0]]), axes=[0, 1])
    debugging.assert_near(m_mean, 4.0)
    debugging.assert_near(m_var, 5.0)
    m_mean_ax, _ = nn.moments(Tensor([1.0, 3.0]), axes=0)
    debugging.assert_near(m_mean_ax, 2.0)

    nm_mean, nm_var = nn.normalize_moments(
        Tensor(2.0), Tensor(4.0), Tensor(10.0), shift=Tensor(1.0)
    )
    debugging.assert_near(nm_mean, 3.0)
    debugging.assert_near(nm_var, 1.0)
    nm_mean_no_shift, _ = nn.normalize_moments(Tensor(2.0), Tensor(4.0), Tensor(10.0), shift=None)
    debugging.assert_near(nm_mean_no_shift, 2.0)

    bn = nn.batch_normalization(
        Tensor([2.0, 4.0]),
        mean=Tensor([1.0, 1.0]),
        variance=Tensor([1.0, 1.0]),
        offset=None,
        scale=None,
        variance_epsilon=0.0,
    )
    debugging.assert_near(bn, [1.0, 3.0])

    lsm = nn.log_softmax(Tensor([0.0, 0.0]))
    debugging.assert_near(tf.math.exp(lsm), [0.5, 0.5])
    sp = nn.softplus(Tensor([0.0]))
    debugging.assert_near(sp, [math.log(2.0)])
    ss = nn.softsign(Tensor([1.0, -1.0]))
    debugging.assert_near(ss, [0.5, -0.5])

    sce = nn.sigmoid_cross_entropy_with_logits(labels=Tensor([1.0, 0.0]), logits=Tensor([0.0, 0.0]))
    debugging.assert_near(sce, [math.log(2.0), math.log(2.0)])
    smce = nn.softmax_cross_entropy_with_logits(
        labels=Tensor([[1.0, 0.0]]), logits=Tensor([[0.0, 0.0]])
    )
    debugging.assert_near(smce, [math.log(2.0)])
    ssmce = nn.sparse_softmax_cross_entropy_with_logits(
        labels=Tensor([0]), logits=Tensor([[0.0, 0.0]])
    )
    debugging.assert_near(ssmce, [math.log(2.0)])
    wce = nn.weighted_cross_entropy_with_logits(
        labels=Tensor([1.0]), logits=Tensor([0.0]), pos_weight=Tensor(2.0)
    )
    debugging.assert_near(wce, [2.0 * math.log(2.0)])


def test_linalg_extended_ops() -> None:
    """Verify `linalg` LinearOperators and extended decompositions/solvers."""
    mat = Tensor([[4.0, 1.0], [1.0, 3.0]])
    vec = Tensor([1.0, 2.0])
    assert tf.linalg.adjoint(mat).shape == (2, 2)
    assert tf.linalg.matrix_transpose(mat, conjugate=True).shape == (2, 2)
    assert tf.linalg.band_part(mat, -1, 0).shape == (2, 2)
    debugging.assert_near(
        tf.linalg.cross(Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0])),
        [0.0, 0.0, 1.0],
    )
    debugging.assert_near(tf.linalg.diag_part(mat), [4.0, 3.0])
    assert tf.linalg.expm(mat).shape == (2, 2)
    assert tf.linalg.logm(mat).shape == (2, 2)
    debugging.assert_near(tf.linalg.global_norm([Tensor([3.0]), Tensor([4.0])]), 5.0)
    debugging.assert_near(tf.linalg.global_norm([]), 0.0)
    assert tf.linalg.logdet(mat).shape == ()

    lu_f, p_f = tf.linalg.lu(mat)
    debugging.assert_near(tf.linalg.lu_reconstruct(lu_f, p_f), mat)
    debugging.assert_near(tf.linalg.lu_matrix_inverse(lu_f, p_f), tf.linalg.inv(mat))
    rhs = Tensor([[5.0], [4.0]])
    debugging.assert_near(tf.linalg.lu_solve(lu_f, p_f, rhs), tf.linalg.solve(mat, rhs))
    lu_s, _ = tf.linalg.lu(Tensor(2.0))
    assert lu_s.shape == ()

    assert tf.linalg.matmul(
        mat, mat, transpose_a=True, transpose_b=True, output_type=tf.float32
    ).shape == (2, 2)
    assert tf.linalg.matmul(mat, mat, adjoint_a=True, adjoint_b=True).shape == (2, 2)
    debugging.assert_near(tf.linalg.matrix_rank(mat), 2)
    debugging.assert_near(tf.linalg.matvec(mat, vec), [6.0, 7.0])
    normed, n_val = tf.linalg.normalize(Tensor([3.0, 4.0]))
    debugging.assert_near(normed, [0.6, 0.8])
    debugging.assert_near(n_val, 5.0)
    debugging.assert_near(tf.linalg.set_diag(mat, Tensor([9.0, 8.0])), [[9.0, 1.0], [1.0, 8.0]])
    sqrt_mat = tf.linalg.sqrtm(Tensor([[5.0, 4.0], [4.0, 5.0]]))
    debugging.assert_near(sqrt_mat, [[2.0, 1.0], [1.0, 2.0]], atol=1e-5)
    debugging.assert_near(tf.linalg.tensor_diag(Tensor([1.0, 2.0])), [[1.0, 0.0], [0.0, 2.0]])
    debugging.assert_near(tf.linalg.tensor_diag_part(mat), [4.0, 3.0])
    debugging.assert_near(tf.linalg.trace(mat), 7.0)
    tril_mat = tf.linalg.band_part(mat, -1, 0)
    assert tf.linalg.triangular_solve(tril_mat, rhs).shape == (2, 1)

    diags_seq = (Tensor([1.0, 0.0]), Tensor([2.0, 3.0]), Tensor([0.0, 1.0]))
    diags_compact = Tensor([[1.0, 0.0], [2.0, 3.0], [0.0, 1.0]])
    assert tf.linalg.tridiagonal_matmul(diags_seq, rhs, diagonals_format="sequence").shape == (2, 1)
    assert tf.linalg.tridiagonal_matmul(diags_compact, rhs, diagonals_format="compact").shape == (
        2,
        1,
    )
    assert tf.linalg.tridiagonal_matmul(mat, rhs, diagonals_format="matrix").shape == (2, 1)
    assert tf.linalg.tridiagonal_solve(diags_seq, rhs).shape == (2, 1)
    assert tf.linalg.tridiagonal_solve(diags_compact, rhs).shape == (2, 1)

    op_full = tf.linalg.LinearOperatorFullMatrix(mat)
    assert op_full.dtype == mat.dtype
    assert op_full.shape == (2, 2)
    assert op_full.to_dense().shape == (2, 2)
    assert op_full.matmul(mat, adjoint=True, adjoint_arg=True).shape == (2, 2)
    assert op_full.matvec(vec).shape == (2,)
    assert op_full.solve(rhs, adjoint=True, adjoint_arg=False).shape == (2, 1)
    assert op_full.determinant().shape == ()
    assert op_full.log_abs_determinant().shape == ()
    assert op_full.trace().shape == ()
    assert isinstance(op_full.inverse(), tf.linalg.LinearOperatorFullMatrix)
    assert isinstance(op_full.cholesky(), tf.linalg.LinearOperatorLowerTriangular)
    assert isinstance(op_full.adjoint(), tf.linalg.LinearOperatorFullMatrix)

    op_diag = tf.linalg.LinearOperatorDiag([2.0, 3.0])
    debugging.assert_near(op_diag.to_dense(), [[2.0, 0.0], [0.0, 3.0]])
    op_id = tf.linalg.LinearOperatorIdentity(2)
    debugging.assert_near(op_id.to_dense(), [[1.0, 0.0], [0.0, 1.0]])
    op_tril = tf.linalg.LinearOperatorLowerTriangular(mat)
    debugging.assert_near(op_tril.to_dense(), [[4.0, 0.0], [1.0, 3.0]])

    assert tf.linalg.band_part(vec, -1, 0).shape == (2,)

    @function
    def traced_linalg_nn(x: Tensor) -> tuple[Tensor, Tensor]:
        """Trace `linalg.lu` and `nn.fractional_max_pool` under `@tf.function`.

        Args:
            x: Input matrix tensor.

        Returns:
            tuple[Tensor, Tensor]: Traced LU factor and pooled tensor.
        """
        lu_t, _ = tf.linalg.lu(x)
        fmp_t, _, _ = nn.fractional_max_pool(x)
        _ = tf.linalg.diag(Tensor([1.0, 2.0]))
        _ = tf.linalg.diag(Tensor([1.0, 2.0]), k=1)
        _ = tf.linalg.lstsq(x, x)
        _ = tf.linalg.set_diag(x, Tensor([1.0, 2.0]))
        return lu_t, fmp_t

    traced_linalg_nn(mat)

    from zero_tensorflow import linalg_ops

    assert linalg_ops._to_nested_floats(Tensor(2.5)) == [[2.5]]
    assert linalg_ops._to_nested_floats(Tensor([2.5, 3.5])) == [[2.5], [3.5]]
    assert linalg_ops._to_1d_floats(Tensor(2.5)) == [2.5]
    assert tf.linalg.einsum("ij,jk->ik", mat, mat, optimize=True).shape == (2, 2)
    assert tf.linalg.norm(vec, ord=1, axis=[0]).shape == ()
    assert tf.linalg.norm(vec, ord="inf").shape == ()
    assert tf.linalg.norm(vec, ord="-inf").shape == ()
    assert tf.linalg.norm(vec, ord=3).shape == ()
    assert tf.linalg.pinv(mat).shape == (2, 2)
    assert len(tf.linalg.slogdet(mat)) == 2
    assert tf.linalg.diag_part(mat, k=(-1, 1)).shape == (3, 2)
    assert tf.linalg.set_diag(
        mat, Tensor([[1.0, 0.0], [9.0, 8.0], [2.0, 0.0]]), k=(-1, 1)
    ).shape == (
        2,
        2,
    )

    assert mat.data is not None
    assert Tensor(None).data is None
    debugging.assert_near(
        tf.math.reduce_sum(Tensor([[1.0, 2.0], [3.0, 4.0]]), axis=(0, 1), name="sum"),
        10.0,
    )
    from zero_tensorflow import math_ops

    assert math_ops._normalize_axis(None) is None
    assert math_ops._normalize_axis(0) == 0
    assert math_ops._normalize_axis(Tensor(1, dtype=tf.int32)) == 1
    assert math_ops._normalize_axis(Tensor([0, 1], dtype=tf.int32)) == (0, 1)
    assert math_ops._normalize_axis([0, 1]) == (0, 1)
    assert math_ops._normalize_axis("invalid") is None

    # Test new LinearOperator* classes and properties
    assert isinstance(op_full.H, tf.linalg.LinearOperatorFullMatrix)
    assert len(op_full.graph_parents) == 1
    assert op_full.parameters["name"] == "LinearOperatorFullMatrix"

    assert tf.linalg.LinearOperatorAdjoint(op_full).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorAdjoint(mat).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorBlockDiag([op_full, mat]).to_dense().shape == (4, 4)
    assert tf.linalg.LinearOperatorBlockDiag([op_full]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorBlockDiag(()).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorBlockLowerTriangular([[op_full]]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorBlockLowerTriangular(()).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorBlockLowerTriangular([[]]).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorCirculant([1.0, 2.0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorCirculant([]).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorCirculant2D([1.0, 2.0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorCirculant3D([1.0, 2.0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorComposition([op_full, op_full]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorComposition(()).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorHouseholder([1.0, 0.0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorHouseholder(1.0).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorInversion(op_full).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorKronecker([op_full, op_full]).to_dense().shape == (4, 4)
    assert tf.linalg.LinearOperatorKronecker([op_full]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorKronecker(()).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorLowRankUpdate(
        op_full, [[1.0], [0.0]], diag_update=[2.0]
    ).to_dense().shape == (
        2,
        2,
    )
    assert tf.linalg.LinearOperatorPermutation([1, 0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorPermutation(0).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorScaledIdentity(2, 3.0).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorToeplitz([1.0, 2.0], [1.0, 3.0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorTridiag(diags_compact).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorZeros(2, 3).to_dense().shape == (2, 3)

    # Specialized LinearOperator methods verification
    debugging.assert_near(op_diag.diag_part(), [2.0, 3.0])
    debugging.assert_near(op_diag.trace(), 5.0)
    debugging.assert_near(op_diag.determinant(), 6.0)
    debugging.assert_near(op_diag.log_abs_determinant(), math.log(6.0))
    debugging.assert_near(op_diag.matmul(rhs), [[10.0], [12.0]])
    debugging.assert_near(op_diag.solve(Tensor([[10.0], [12.0]])), rhs)

    debugging.assert_near(op_id.diag_part(), [1.0, 1.0])
    debugging.assert_near(op_id.trace(), 2.0)
    debugging.assert_near(op_id.determinant(), 1.0)
    debugging.assert_near(op_id.log_abs_determinant(), 0.0)
    debugging.assert_near(op_id.matmul(rhs), rhs)
    debugging.assert_near(op_id.solve(rhs), rhs)

    debugging.assert_near(op_tril.diag_part(), [4.0, 3.0])
    debugging.assert_near(op_tril.determinant(), 12.0)
    debugging.assert_near(op_tril.log_abs_determinant(), math.log(12.0))
    debugging.assert_near(op_tril.solve(op_tril.matmul(rhs)), rhs)

    op_adj = tf.linalg.LinearOperatorAdjoint(op_full)
    debugging.assert_near(op_adj.matmul(rhs), op_full.matmul(rhs, adjoint=True))
    debugging.assert_near(op_adj.solve(rhs), op_full.solve(rhs, adjoint=True))
    op_adj_raw = tf.linalg.LinearOperatorAdjoint(mat)
    debugging.assert_near(op_adj_raw.matmul(rhs), op_full.matmul(rhs, adjoint=True))
    debugging.assert_near(op_adj_raw.solve(rhs), op_full.solve(rhs, adjoint=True))

    op_bdiag = tf.linalg.LinearOperatorBlockDiag([op_diag, mat])
    debugging.assert_near(op_bdiag.determinant(), 6.0 * 11.0)
    debugging.assert_near(op_bdiag.trace(), 5.0 + 7.0)
    assert tf.linalg.LinearOperatorBlockDiag(()).determinant().numpy() == 1.0
    assert tf.linalg.LinearOperatorBlockDiag(()).trace().numpy() == 1.0

    op_circ = tf.linalg.LinearOperatorCirculant([2.0, 4.0])
    debugging.assert_near(op_circ.determinant(), 8.0)
    debugging.assert_near(op_circ.trace(), 6.0)

    op_comp = tf.linalg.LinearOperatorComposition([op_diag, mat])
    debugging.assert_near(op_comp.matmul(rhs), op_diag.matmul(op_full.matmul(rhs)))
    debugging.assert_near(
        op_comp.matmul(rhs, adjoint=True),
        op_full.matmul(op_diag.matmul(rhs, adjoint=True), adjoint=True),
    )
    debugging.assert_near(tf.linalg.LinearOperatorComposition(()).matmul(rhs), rhs)

    op_hh = tf.linalg.LinearOperatorHouseholder([1.0, 0.0])
    debugging.assert_near(op_hh.determinant(), -1.0)
    debugging.assert_near(op_hh.log_abs_determinant(), 0.0)
    debugging.assert_near(op_hh.solve(rhs), op_hh.matmul(rhs))

    op_inv = tf.linalg.LinearOperatorInversion(op_full)
    debugging.assert_near(op_inv.matmul(rhs), op_full.solve(rhs))
    debugging.assert_near(op_inv.solve(rhs), op_full.matmul(rhs))
    op_inv_raw = tf.linalg.LinearOperatorInversion(mat)
    debugging.assert_near(op_inv_raw.matmul(rhs), op_full.solve(rhs))
    debugging.assert_near(op_inv_raw.solve(rhs), op_full.matmul(rhs))

    op_kron = tf.linalg.LinearOperatorKronecker([op_diag, mat])
    debugging.assert_near(op_kron.trace(), 5.0 * 7.0)
    assert tf.linalg.LinearOperatorKronecker(()).trace().numpy() == 1.0

    op_lru = tf.linalg.LinearOperatorLowRankUpdate(op_diag, [[1.0], [0.0]], diag_update=[2.0])
    debugging.assert_near(op_lru.matmul(rhs), op_lru.to_dense() @ rhs)
    op_lru_raw = tf.linalg.LinearOperatorLowRankUpdate(mat, [[1.0], [0.0]])
    debugging.assert_near(op_lru_raw.matmul(rhs), op_lru_raw.to_dense() @ rhs)

    op_perm = tf.linalg.LinearOperatorPermutation([1, 0])
    debugging.assert_near(op_perm.matmul(rhs), [[4.0], [5.0]])
    debugging.assert_near(op_perm.matmul(rhs, adjoint=True), [[4.0], [5.0]])
    debugging.assert_near(op_perm.log_abs_determinant(), 0.0)
    assert tf.linalg.LinearOperatorPermutation(0).matmul(Tensor([[3.0]])).shape == (1, 1)

    op_sc = tf.linalg.LinearOperatorScaledIdentity(2, 3.0)
    debugging.assert_near(op_sc.diag_part(), [3.0, 3.0])
    debugging.assert_near(op_sc.trace(), 6.0)
    debugging.assert_near(op_sc.determinant(), 9.0)
    debugging.assert_near(op_sc.matmul(rhs), rhs * 3.0)
    debugging.assert_near(op_sc.solve(rhs * 3.0), rhs)

    op_toep = tf.linalg.LinearOperatorToeplitz([1.0, 2.0], [1.0, 3.0])
    debugging.assert_near(op_toep.to_dense(), [[1.0, 3.0], [2.0, 1.0]])
    op_trid = tf.linalg.LinearOperatorTridiag(diags_compact)
    assert op_trid.to_dense().shape == (2, 2)

    op_z = tf.linalg.LinearOperatorZeros(2, 2)
    debugging.assert_near(op_z.determinant(), 0.0)
    debugging.assert_near(op_z.trace(), 0.0)
    debugging.assert_near(op_z.matmul(rhs), [[0.0], [0.0]])
    assert op_full.diag_part().shape == (2,)

    # Batched 3-D linalg decompositions & functions
    b_mat = Tensor([[[4.0, 1.0], [1.0, 3.0]], [[2.0, 0.0], [0.0, 5.0]]])
    b_rhs = Tensor([[[5.0], [4.0]], [[4.0], [10.0]]])
    assert tf.linalg.band_part(b_mat, 0, 0).shape == (2, 2, 2)
    assert tf.linalg.cholesky(b_mat).shape == (2, 2, 2)
    assert tf.linalg.diag(Tensor([[1.0, 2.0], [3.0, 4.0]]), k=1, num_rows=3, num_cols=3).shape == (
        2,
        3,
        3,
    )
    assert tf.linalg.diag(Tensor([1.0, 2.0]), num_rows=3, num_cols=3).shape == (3, 3)
    assert tf.linalg.matrix_transpose(mat, conjugate=False).shape == (2, 2)
    assert tf.linalg.set_diag(mat, Tensor([1.0]), k=1).shape == (2, 2)
    assert tf.linalg.diag_part(b_mat, k=(-1, 1)).shape == (2, 3, 2)
    assert tf.linalg.diag_part(b_mat, k=0).shape == (2, 2)
    e_vals, e_vecs = tf.linalg.eigh(b_mat)
    assert e_vals.shape == (2, 2) and e_vecs.shape == (2, 2, 2)
    assert tf.linalg.expm(b_mat).shape == (2, 2, 2)
    assert tf.linalg.eye(2, batch_shape=(2,)).shape == (2, 2, 2)
    assert tf.linalg.inv(b_mat).shape == (2, 2, 2)
    assert tf.linalg.lstsq(b_mat, b_rhs).shape == (2, 2, 1)
    assert tf.linalg.lstsq(mat, rhs, l2_regularizer=0.1).shape == (2, 1)
    b_lu, b_p = tf.linalg.lu(b_mat)
    assert b_lu.shape == (2, 2, 2) and b_p.shape == (2, 2)
    p_mat = Tensor([[0.0, 2.0], [3.0, 1.0]])
    p_lu, p_perm = tf.linalg.lu(p_mat)
    debugging.assert_near(tf.linalg.lu_reconstruct(p_lu, p_perm), p_mat)
    b_q, b_r = tf.linalg.qr(b_mat, full_matrices=True)
    assert b_q.shape == (2, 2, 2) and b_r.shape == (2, 2, 2)
    q_2d, r_2d = tf.linalg.qr(mat, full_matrices=False)
    assert q_2d.shape == (2, 2) and r_2d.shape == (2, 2)
    s_2d, u_2d, v_2d = tf.linalg.svd(mat, compute_uv=True)
    assert s_2d.shape == (2,) and u_2d.shape == (2, 2) and v_2d.shape == (2, 2)
    assert tf.linalg.set_diag(b_mat, Tensor([[9.0, 8.0], [7.0, 6.0]])).shape == (2, 2, 2)
    assert tf.linalg.set_diag(
        b_mat,
        Tensor([[[1.0, 0.0], [9.0, 8.0], [2.0, 0.0]], [[1.0, 0.0], [7.0, 6.0], [2.0, 0.0]]]),
        k=(-1, 1),
    ).shape == (2, 2, 2)
    assert tf.linalg.solve(b_mat, b_rhs).shape == (2, 2, 1)
    assert tf.linalg.sqrtm(b_mat).shape == (2, 2, 2)
    b_s, b_u, b_v = tf.linalg.svd(b_mat, full_matrices=True, compute_uv=True)
    assert b_s.shape == (2, 2) and b_u.shape == (2, 2, 2) and b_v.shape == (2, 2, 2)
    b_s_only = tf.linalg.svd(b_mat, compute_uv=False)
    assert isinstance(b_s_only, Tensor) and b_s_only.shape == (2, 2)
    s_only = tf.linalg.svd(mat, compute_uv=False)
    assert isinstance(s_only, Tensor) and s_only.shape == (2,)
    assert tf.linalg.trace(b_mat).shape == (2,)
    assert tf.linalg.triangular_solve(tf.linalg.band_part(b_mat, -1, 0), b_rhs).shape == (2, 2, 1)
    assert tf.linalg.l2_normalize(Tensor([[3.0, 4.0]]), axis=(0, 1)).shape == (1, 2)

    # Test new linalg solver and decomposition methods
    assert tf.linalg.banded_triangular_solve(tril_mat, rhs).shape == (2, 1)
    chol = tf.linalg.cholesky(mat)
    assert tf.linalg.cholesky_solve(chol, rhs).shape == (2, 1)
    assert tf.linalg.diag([1.0, 2.0], k=1).shape == (3, 3)
    assert tf.linalg.eig(mat) is not None
    assert tf.linalg.eigvals(mat).shape == (2,)
    eig_t1 = tf.linalg.eigh_tridiagonal([2.0, 3.0], [1.0], eigvals_only=True)
    assert isinstance(eig_t1, Tensor) and eig_t1.shape == (2,)
    assert isinstance(tf.linalg.eigh_tridiagonal([2.0, 3.0], [1.0], eigvals_only=False), tuple)
    eig_t2 = tf.linalg.eigh_tridiagonal([2.0], [], eigvals_only=True)
    assert isinstance(eig_t2, Tensor) and eig_t2.shape == (1,)
    assert tf.linalg.eye(2, 3).shape == (2, 3)
    assert tf.linalg.inv(mat, adjoint=True).shape == (2, 2)
    assert tf.linalg.l2_normalize(vec).shape == (2,)
    assert tf.linalg.norm(vec, ord=2, axis=0, keepdims=True).shape == (1,)
    assert tf.linalg.solve(mat, rhs, adjoint=True).shape == (2, 1)

    # Test new tf.nn wrappers and functions
    def _dummy_cell(inp: TensorOperand, st: TensorOperand) -> tuple[Tensor, TensorOperand]:
        """Dummy RNN cell returning `(inp * 2, st)`.

        Args:
            inp: Step input.
            st: Step state.

        Returns:
            tuple[Tensor, TensorOperand]: Output and state.
        """
        return (Tensor(inp) * 2.0, st)

    def _dummy_cell_single(inp: TensorOperand, st: TensorOperand) -> Tensor:
        """Dummy RNN cell returning single `Tensor`.

        Args:
            inp: Step input.
            st: Step state.

        Returns:
            Tensor: Output tensor.
        """
        del st
        return Tensor(inp) + 1.0

    w_dev = nn.RNNCellDeviceWrapper(cell=_dummy_cell, device="/cpu:0")
    assert w_dev.device == "/cpu:0"
    out_dev, _ = w_dev(Tensor([2.0]), Tensor([1.0]))
    debugging.assert_near(out_dev, [4.0])
    assert nn.RNNCellDeviceWrapper(cell=_dummy_cell_single)(Tensor([2.0]), Tensor([1.0]))[
        0
    ].numpy() == [3.0]
    assert nn.RNNCellDeviceWrapper(cell=None)(Tensor([2.0]), Tensor([1.0]))[0].numpy() == [2.0]

    w_drop = nn.RNNCellDropoutWrapper(
        cell=_dummy_cell,
        input_keep_prob=0.8,
        output_keep_prob=0.8,
        state_keep_prob=0.8,
        seed=42,
    )
    assert w_drop.input_keep_prob == 0.8
    assert w_drop(Tensor([2.0]), Tensor([1.0]))[0].shape == (1,)
    assert nn.RNNCellDropoutWrapper(cell=_dummy_cell_single)(Tensor([2.0]), Tensor([1.0]))[
        0
    ].shape == (1,)
    assert nn.RNNCellDropoutWrapper(cell=None)(Tensor([2.0]), None)[0].shape == (1,)

    w_res = nn.RNNCellResidualWrapper(cell=_dummy_cell)
    debugging.assert_near(w_res(Tensor([2.0]), Tensor([1.0]))[0], [6.0])
    w_res_fn = nn.RNNCellResidualWrapper(
        cell=_dummy_cell_single,
        residual_fn=lambda i, o: Tensor(i) + Tensor(o) * 2.0,
    )
    debugging.assert_near(w_res_fn(Tensor([2.0]), Tensor([1.0]))[0], [8.0])
    assert nn.RNNCellResidualWrapper(cell=None)(Tensor([2.0]), Tensor([1.0]))[0].numpy() == [4.0]

    # NCHW / NCDHW conv2d and conv3d tests
    x_nchw = tf.ones((1, 2, 4, 4), dtype=tf.float32)
    f_nchw = tf.ones((2, 2, 2, 3), dtype=tf.float32)
    assert nn.conv2d(
        x_nchw,
        f_nchw,
        strides=[1, 1, 2, 2],
        padding="SAME",
        data_format="NCHW",
        dilations=[1, 1, 1, 1],
    ).shape == (1, 3, 2, 2)
    x_ncdhw = tf.ones((1, 2, 3, 3, 3), dtype=tf.float32)
    f_ncdhw = tf.ones((2, 2, 2, 2, 4), dtype=tf.float32)
    assert nn.conv3d(
        x_ncdhw,
        f_ncdhw,
        strides=[1, 1, 1, 1, 1],
        padding="SAME",
        data_format="NCDHW",
        dilations=[1, 1, 1, 1, 1],
    ).shape == (1, 4, 3, 3, 3)

    assert len(nn.all_candidate_sampler([[0]], 1, 2, True)) == 3
    x4d = tf.ones((1, 2, 2, 2), dtype=tf.float32)
    f4d = tf.ones((2, 2, 2, 2), dtype=tf.float32)
    assert nn.atrous_conv2d(x4d, f4d, rate=1, padding="SAME").shape == (1, 2, 2, 2)
    assert isinstance(
        nn.atrous_conv2d_transpose(x4d, f4d, [1, 2, 2, 2], rate=1, padding="VALID"),
        Tensor,
    )
    debugging.assert_near(nn.bias_add(Tensor([1.0, 2.0]), Tensor([0.5, 1.5])), [1.5, 3.5])
    assert nn.bias_add(tf.ones((1, 2, 3, 3)), Tensor([1.0, 2.0]), data_format="NCHW").shape == (
        1,
        2,
        3,
        3,
    )
    assert len(nn.collapse_repeated([[1, 1, 2]], [3])) == 2
    assert len(nn.compute_accidental_hits([[0]], [0, 1], 1)) == 3
    debugging.assert_near(nn.compute_average_loss(Tensor([2.0, 4.0]), sample_weight=1.0), 3.0)
    debugging.assert_near(nn.compute_average_loss(Tensor(4.0), global_batch_size=2), 2.0)

    x3d_c = Tensor(
        [
            [[0.1, 5.0, 0.2]],
            [[0.1, 5.0, 0.2]],
            [[0.1, 0.2, 6.0]],
            [[4.0, 0.1, 0.2]],
        ]
    )
    f3d_c = tf.ones((2, 2, 2), dtype=tf.float32)
    assert isinstance(
        nn.conv1d_transpose(x4d, f4d, [1, 2, 2, 2], strides=1, padding="VALID"), Tensor
    )
    assert isinstance(nn.conv_transpose(x4d, f4d, [1, 2, 2, 2], strides=1, padding="VALID"), Tensor)
    assert isinstance(nn.convolution(tf.ones((1, 2, 2)), f3d_c), Tensor)

    dec_beam, log_p = nn.ctc_beam_search_decoder(x3d_c, [4], beam_width=2, top_paths=2)
    assert len(dec_beam) == 2 and log_p.shape == (1, 2)
    dec_greedy, _ = nn.ctc_greedy_decoder(x3d_c, [4], merge_repeated=True)
    debugging.assert_equal(dec_greedy[0].values, [1, 0])
    assert len(nn.ctc_greedy_decoder(None)) == 2
    assert len(nn.ctc_greedy_decoder(Tensor([1.0, 2.0]))) == 2
    assert len(nn.ctc_unique_labels([[1, 2]])) == 2
    assert nn.depthwise_conv2d_backprop_filter(x4d, (2, 2, 2, 1), x4d).shape == (2, 2, 2, 1)
    assert nn.depthwise_conv2d_backprop_filter(None, (2, 2, 2, 1), None).shape == (2, 2, 2, 1)
    assert nn.depthwise_conv2d_backprop_input((1, 4, 4, 2), f4d, x4d).shape == (1, 4, 4, 2)
    assert nn.depthwise_conv2d_backprop_input((1, 4, 4, 2), None, None).shape == (1, 4, 4, 2)
    assert nn.dilation2d(x4d, tf.ones((2, 2, 2)), padding="SAME").shape == (1, 2, 2, 2)
    assert nn.erosion2d(x4d, tf.ones((2, 2, 2)), padding="VALID").shape == (1, 1, 1, 2)
    assert nn.dilation2d(None, None).numpy() == 0.0
    assert nn.dilation2d(Tensor([1.0]), None).shape == (1,)
    assert nn.dilation2d(x4d, Tensor([1.0])).shape == (1, 2, 2, 2)
    assert isinstance(nn.embedding_lookup_sparse(Tensor([[1.0, 2.0]]), Tensor([0])), Tensor)
    assert isinstance(nn.embedding_lookup_sparse(None, Tensor([0])), Tensor)
    assert isinstance(nn.safe_embedding_lookup_sparse(Tensor([[1.0, 2.0]]), Tensor([0])), Tensor)
    assert len(nn.fixed_unigram_candidate_sampler([[0]], 1, 2, True, 5)) == 3
    assert len(nn.learned_unigram_candidate_sampler([[0]], 1, 2, True, 5)) == 3
    fap_out, _, _ = nn.fractional_avg_pool(
        tf.ones((1, 4, 4, 1)), [1.0, 1.5, 1.5, 1.0], overlapping=True
    )
    assert fap_out.shape == (1, 2, 2, 1)
    assert nn.fractional_avg_pool(Tensor([1.0]))[0].shape == (1,)
    with pytest.raises(NotImplementedError):
        nn.fractional_avg_pool(None)
    iso_y, iso_seg = nn.isotonic_regression(Tensor([1.0, 3.0, 2.0]), decreasing=True)
    debugging.assert_near(iso_y, [2.0, 2.0, 2.0])
    assert iso_seg.shape == (3,)
    iso_2d, _ = nn.isotonic_regression(Tensor([[3.0, 1.0, 2.0]]), decreasing=False)
    debugging.assert_near(iso_2d, [[2.0, 2.0, 2.0]])
    assert nn.isotonic_regression(None)[0].numpy() == 0.0
    assert nn.isotonic_regression(Tensor(1.0))[0].numpy() == 1.0
    assert nn.isotonic_regression(x4d)[0].shape == (1, 2, 2, 2)
    debugging.assert_near(nn.l2_loss(Tensor([2.0, 4.0])), 10.0)
    assert isinstance(nn.log_poisson_loss(Tensor([1.0]), Tensor([0.0])), Tensor)
    val_p, idx_p = nn.max_pool_with_argmax(
        Tensor([[[[1.0, 2.0], [5.0, 1.0]], [[3.0, 4.0], [2.0, 8.0]]]]),
        ksize=2,
        strides=2,
        padding="VALID",
        include_batch_in_index=True,
    )
    debugging.assert_near(val_p, [[[[5.0, 8.0]]]])
    debugging.assert_equal(idx_p, [[[[2, 7]]]])
    assert nn.max_pool_with_argmax(x4d, ksize=2, strides=1, padding="SAME")[0].shape == (1, 2, 2, 2)
    assert nn.max_pool_with_argmax(Tensor([[[1.0, 2.0], [3.0, 4.0]]]))[0].shape == (1, 1, 2)
    assert nn.pool(x4d, (2, 2), "AVG", strides=(2, 2), padding="VALID").shape == (1, 1, 1, 2)
    assert nn.pool(x4d, (2, 2), "MAX", strides=(2, 2), padding="VALID").shape == (1, 1, 1, 2)
    debugging.assert_near(nn.relu6(Tensor([-1.0, 3.0, 8.0])), [0.0, 3.0, 6.0])
    debugging.assert_near(nn.scale_regularization_loss(2.5), 2.5)
    cnt_s, mss_s, vss_s, sh_s = nn.sufficient_statistics(Tensor([1.0, 2.0]), axes=0, shift=1.0)
    assert cnt_s.shape == () and mss_s.shape == () and vss_s.shape == ()
    assert isinstance(sh_s, Tensor)
    _, _, _, sh_none = nn.sufficient_statistics(Tensor([[1.0, 2.0]]), axes=[0, 1])
    assert sh_none is None
    wm_m, wm_v = nn.weighted_moments(
        Tensor([1.0, 3.0]), axes=0, frequency_weights=Tensor([1.0, 1.0])
    )
    assert wm_m.shape == () and wm_v.shape == ()
    assert isinstance(nn.with_space_to_batch(x4d, 1, "VALID", None), Tensor)
    assert nn.with_space_to_batch(None).numpy() == 0.0
    s2b_res = nn.with_space_to_batch(
        x4d,
        dilation_rate=[1, 1],
        padding="SAME",
        op=lambda padded, num_sp, pad_mode: padded,
        filter_shape=[2, 2],
    )
    assert s2b_res.shape == (1, 2, 2, 2)
    s2b_val = nn.with_space_to_batch(
        x4d,
        dilation_rate=1,
        padding="VALID",
        op=lambda padded, num_sp, pad_mode: padded,
    )
    assert s2b_val.shape == (1, 2, 2, 2)
    assert tf.zeros((2,), dtype="float32").dtype == tf.float32
