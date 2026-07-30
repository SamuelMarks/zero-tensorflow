import tensorflow as tf
from tensorflow.python.eager import def_function
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class DefFunctionTest(test_util.TensorFlowTestCase):
    def testBasic(self):
        @def_function.function
        def add(a, b):
            return a + b

        # In zero-tensorflow, function tracing currently produces a LogicalNode/ProxyTensor tuple.
        # This just ensures we can trace it.
        res = add(tf.Tensor(1.0), tf.Tensor(2.0))
        # It returns a Tensor object tracing the result
        self.assertTrue(isinstance(res, tf.Tensor))


if __name__ == "__main__":
    test.main()
