import numpy as np
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class NNOpsTest(test_util.TensorFlowTestCase):
    def testRelu(self):
        x = np.array([[-1.0, 2.0], [3.0, -4.0]])
        res = nn_ops.relu(x)
        self.assertAllClose(res, np.array([[0.0, 2.0], [3.0, 0.0]]))

    def testSoftmax(self):
        x = np.array([[1.0, 1.0]])
        res = nn_ops.softmax(x)
        self.assertAllClose(res, np.array([[0.5, 0.5]]))

    def assertAllClose(self, a, b):
        a = a.numpy() if hasattr(a, "numpy") else a
        np.testing.assert_allclose(a, b)


if __name__ == "__main__":
    test.main()
