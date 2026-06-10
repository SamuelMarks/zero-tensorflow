import tensorflow as tf
from tensorflow.python.framework import test_util
import numpy as np
import unittest


@test_util.run_all_in_graph_and_eager_modes
class ReduceTest(test_util.TensorFlowTestCase):
    def testReduceSum(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        with test_util.device(use_gpu=True):
            y_tf = self.evaluate(tf.math.reduce_sum(x))
            self.assertEqual(y_tf, 21)


if __name__ == "__main__":
    unittest.main()
