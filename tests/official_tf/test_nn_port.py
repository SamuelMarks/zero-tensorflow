"""Ported `nn_ops` (`relu`, `softmax`) unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase
from zero_tensorflow import nn as nn_ops


class NNOpsTest(TensorFlowTestCase):
    """Ported neural network operations test case."""

    def testRelu(self) -> None:
        """Verify `nn_ops.relu` on a 2D array with negative and positive values."""
        x = tf.Tensor([[-1.0, 2.0], [3.0, -4.0]])
        res = nn_ops.relu(x)
        self.assertAllClose(res, [[0.0, 2.0], [3.0, 0.0]])
        self.assertDifferentialParity(lambda backend: backend.nn.relu(x))

    def testSoftmax(self) -> None:
        """Verify `nn_ops.softmax` on uniform 2-class logits."""
        x = tf.Tensor([[1.0, 1.0]])
        res = nn_ops.softmax(x)
        self.assertAllClose(res, [[0.5, 0.5]])
        self.assertDifferentialParity(lambda backend: backend.nn.softmax(x))

    def testImageNmsAndCropResize(self) -> None:
        """Verify `image.non_max_suppression` and `image.crop_and_resize` on multi-box fixtures."""
        boxes = tf.Tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.05, 1.0, 1.05], [0.5, 0.5, 1.0, 1.0]])
        scores = tf.Tensor([0.9, 0.85, 0.7])
        sel = tf.image.non_max_suppression(boxes, scores, max_output_size=2, iou_threshold=0.5)
        self.assertAllClose(sel, [0, 2])

        img = tf.ones((1, 4, 4, 1), dtype=tf.float32)
        cropped = tf.image.crop_and_resize(img, [[0.0, 0.0, 0.5, 0.5]], [0], [2, 2])
        self.assertEqual(cropped.shape, (1, 2, 2, 1))

    def testBatchedLinalgAndOperators(self) -> None:
        """Verify batched `linalg.solve`, `qr`, `svd`, `cholesky`, `expm`, and `LinearOperator`s."""
        b_mat = tf.Tensor([[[4.0, 1.0], [1.0, 3.0]], [[2.0, 0.0], [0.0, 5.0]]])
        b_rhs = tf.Tensor([[[5.0], [4.0]], [[4.0], [10.0]]])
        sol = tf.linalg.solve(b_mat, b_rhs)
        self.assertAllClose(sol, [[[1.0], [1.0]], [[2.0], [2.0]]])
        q, r = tf.linalg.qr(b_mat)
        self.assertAllClose(tf.linalg.matmul(q, r), b_mat)
        chol = tf.linalg.cholesky(b_mat)
        self.assertAllClose(tf.linalg.matmul(chol, chol, adjoint_b=True), b_mat)
        s, u, v = tf.linalg.svd(b_mat)
        self.assertEqual(s.shape, (2, 2))
        self.assertEqual(u.shape, (2, 2, 2))
        self.assertEqual(v.shape, (2, 2, 2))
        self.assertEqual(tf.linalg.expm(b_mat).shape, (2, 2, 2))
        op_d = tf.linalg.LinearOperatorDiag([2.0, 4.0])
        self.assertAllClose(op_d.determinant(), 8.0)
