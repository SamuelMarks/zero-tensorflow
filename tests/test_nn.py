"""Unit tests for `test_nn`."""

from __future__ import annotations

import math

import zero_tensorflow as tf
from zero_tensorflow import Tensor, debugging, function, nn

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
    pt_k = tf.ones((1, 1, 2, 3), dtype=tf.float32)
    assert isinstance(nn.separable_conv2d(x2d_hwc, dw_k, pt_k), Tensor)

    crelu_out = nn.crelu(Tensor([-1.0, 2.0]))
    debugging.assert_near(crelu_out, [0.0, 2.0, 1.0, 0.0])

    assert isinstance(nn.gelu(Tensor([0.0, 1.0])), Tensor)
    assert isinstance(nn.silu(Tensor([0.0, 1.0])), Tensor)
    assert isinstance(nn.swish(Tensor([0.0, 1.0])), Tensor)

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
        return lu_t, fmp_t

    traced_linalg_nn(mat)

    assert mat.data is not None
    assert Tensor(None).data is None
    debugging.assert_near(
        tf.math.reduce_sum(Tensor([[1.0, 2.0], [3.0, 4.0]]), axis=(0, 1), name="sum"),
        10.0,
    )
    from zero_tensorflow import math_ops

    assert math_ops._clean_kwargs(
        {"dim": (0, "skip"), "empty": ("skip",), "dtype": tf.float32, "other": Tensor(1.0)}
    ) == {"axis": (0,), "dtype": "float32"}

    # Test new LinearOperator* classes and properties
    assert isinstance(op_full.H, tf.linalg.LinearOperatorFullMatrix)
    assert len(op_full.graph_parents) == 1
    assert op_full.parameters["name"] == "LinearOperatorFullMatrix"

    assert tf.linalg.LinearOperatorAdjoint(op_full).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorAdjoint(mat).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorBlockDiag([op_full, mat]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorBlockDiag(()).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorBlockLowerTriangular([[op_full]]).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorCirculant([1.0, 2.0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorCirculant2D([1.0, 2.0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorCirculant3D([1.0, 2.0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorComposition([op_full, op_full]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorComposition(()).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorHouseholder([1.0, 0.0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorHouseholder(1.0).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorInversion(op_full).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorKronecker([op_full]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorKronecker(()).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorLowRankUpdate(op_full, [[1.0], [0.0]]).to_dense().shape == (
        2,
        2,
    )
    assert tf.linalg.LinearOperatorPermutation([1, 0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorPermutation(0).to_dense().shape == (1, 1)
    assert tf.linalg.LinearOperatorScaledIdentity(2, 3.0).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorToeplitz([1.0, 2.0], [1.0, 3.0]).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorTridiag(diags_compact).to_dense().shape == (2, 2)
    assert tf.linalg.LinearOperatorZeros(2, 3).to_dense().shape == (2, 3)

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
    w_dev = nn.RNNCellDeviceWrapper(cell="c", device="/cpu:0")
    assert w_dev.device == "/cpu:0"
    w_drop = nn.RNNCellDropoutWrapper(cell="c", input_keep_prob=0.8)
    assert w_drop.input_keep_prob == 0.8
    w_res = nn.RNNCellResidualWrapper(cell="c")
    assert w_res.cell == "c"

    assert len(nn.all_candidate_sampler([[0]], 1, 2, True)) == 3
    x4d = tf.ones((1, 2, 2, 2), dtype=tf.float32)
    f4d = tf.ones((2, 2, 2, 2), dtype=tf.float32)
    assert nn.atrous_conv2d(x4d, f4d, rate=1, padding="SAME").shape == (1, 2, 2, 2)
    assert isinstance(
        nn.atrous_conv2d_transpose(x4d, f4d, [1, 2, 2, 2], rate=1, padding="VALID"),
        Tensor,
    )
    debugging.assert_near(nn.bias_add(Tensor([1.0, 2.0]), Tensor([0.5, 1.5])), [1.5, 3.5])
    assert len(nn.collapse_repeated([[1, 1, 2]], [3])) == 2
    assert len(nn.compute_accidental_hits([[0]], [0, 1], 1)) == 3
    debugging.assert_near(nn.compute_average_loss(Tensor([2.0, 4.0]), sample_weight=1.0), 3.0)
    debugging.assert_near(nn.compute_average_loss(Tensor(4.0), global_batch_size=2), 2.0)

    x3d_c = tf.ones((1, 2, 2), dtype=tf.float32)
    f3d_c = tf.ones((2, 2, 2), dtype=tf.float32)
    assert isinstance(
        nn.conv1d_transpose(x4d, f4d, [1, 2, 2, 2], strides=1, padding="VALID"), Tensor
    )
    assert isinstance(nn.conv_transpose(x4d, f4d, [1, 2, 2, 2], strides=1, padding="VALID"), Tensor)
    assert isinstance(nn.convolution(x3d_c, f3d_c), Tensor)

    assert len(nn.ctc_beam_search_decoder(x3d_c, [4], beam_width=2, top_paths=1)) == 2
    assert len(nn.ctc_greedy_decoder(x3d_c, [4])) == 2
    assert len(nn.ctc_unique_labels([[1, 2]])) == 2
    assert nn.depthwise_conv2d_backprop_filter(x4d, (2, 2, 2, 1), x4d).shape == (2, 2, 2, 1)
    assert nn.depthwise_conv2d_backprop_input((1, 4, 4, 2), f4d, x4d).shape == (1, 4, 4, 2)
    assert isinstance(nn.dilation2d(x4d, tf.ones((2, 2, 2))), Tensor)
    assert isinstance(nn.erosion2d(x4d, tf.ones((2, 2, 2))), Tensor)
    assert isinstance(nn.embedding_lookup_sparse(Tensor([[1.0, 2.0]]), Tensor([0])), Tensor)
    assert isinstance(nn.embedding_lookup_sparse(None, Tensor([0])), Tensor)
    assert isinstance(nn.safe_embedding_lookup_sparse(Tensor([[1.0, 2.0]]), Tensor([0])), Tensor)
    assert len(nn.fixed_unigram_candidate_sampler([[0]], 1, 2, True, 5)) == 3
    assert len(nn.learned_unigram_candidate_sampler([[0]], 1, 2, True, 5)) == 3
    assert len(nn.fractional_avg_pool(x4d, [1.0, 1.5, 1.5, 1.0])) == 3
    assert len(nn.isotonic_regression(Tensor([3.0, 1.0, 2.0]))) == 2
    debugging.assert_near(nn.l2_loss(Tensor([2.0, 4.0])), 10.0)
    assert isinstance(nn.log_poisson_loss(Tensor([1.0]), Tensor([0.0])), Tensor)
    val_p, idx_p = nn.max_pool_with_argmax(x4d, ksize=2, strides=2, padding="VALID")
    assert val_p.shape == (1, 1, 1, 2) and idx_p.shape == (1, 1, 1, 2)
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
    assert tf.zeros((2,), dtype="float32").dtype == tf.float32
